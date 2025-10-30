# vqvae.py
import torch
import torch.nn as nn
import torch.nn.functional as F


# -------------------------
# Helper: Conv block
# -------------------------
class ConvBlock(nn.Module):
    """
    Simple Conv -> (BatchNorm) -> Activation block.
    """
    def __init__(self, in_ch, out_ch, kernel=3, stride=1, padding=1,
                 use_batchnorm=False, activation=nn.ReLU):
        super().__init__()
        layers = [nn.Conv2d(in_ch, out_ch, kernel, stride=stride, padding=padding)]
        if use_batchnorm:
            layers.append(nn.BatchNorm2d(out_ch))
        layers.append(activation(inplace=True))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# -------------------------
# Encoder
# -------------------------
class Encoder(nn.Module):
    """
    Configurable encoder.
    Assumes input shape (B, C, H, W). Downsamples `n_down` times (stride=2 convs)
    and optionally applies `num_residual_blocks` residual convs at the end.
    """
    def __init__(self,
                 in_channels=1,
                 hidden_channels=128,
                 n_down=4,
                 kernel_size=4,
                 use_batchnorm=False,
                 activation=nn.ReLU,
                 num_residual_blocks=0,
                 residual_channels=None):
        super().__init__()

        # initial conv (no downsample)
        self.initial = ConvBlock(in_channels, hidden_channels // 2,
                                 kernel=3, stride=1, padding=1,
                                 use_batchnorm=use_batchnorm,
                                 activation=activation)

        # increase to hidden_channels
        self.conv_increase = ConvBlock(hidden_channels // 2, hidden_channels,
                                       kernel=3, stride=1, padding=1,
                                       use_batchnorm=use_batchnorm,
                                       activation=activation)

        # downsampling blocks (stride=2 conv reduces H,W by 2)
        downs = []
        in_ch = hidden_channels
        for i in range(n_down):
            out_ch = min(hidden_channels * (2 if i > 0 else 1), 4 * hidden_channels)
            # use kernel_size and stride=2 for downsampling
            downs.append(ConvBlock(in_ch, out_ch,
                                   kernel=kernel_size, stride=2,
                                   padding=(kernel_size // 2 - 1),
                                   use_batchnorm=use_batchnorm,
                                   activation=activation))
            in_ch = out_ch
        self.downs = nn.Sequential(*downs)

        # optional residual blocks to increase expressivity
        if num_residual_blocks > 0:
            rc = residual_channels or in_ch
            res_blocks = []
            for _ in range(num_residual_blocks):
                res_blocks.append(nn.Sequential(
                    ConvBlock(in_ch, rc, kernel=3, stride=1, padding=1,
                              use_batchnorm=use_batchnorm, activation=activation),
                    nn.Conv2d(rc, in_ch, kernel_size=1, stride=1, padding=0)
                ))
            self.res_blocks = nn.ModuleList(res_blocks)
        else:
            self.res_blocks = None

    def forward(self, x):
        x = self.initial(x)
        x = self.conv_increase(x)
        x = self.downs(x)
        if self.res_blocks is not None:
            for r in self.res_blocks:
                res = r(x)
                x = x + res  # simple residual
        return x


# -------------------------
# Vector Quantizer
# -------------------------
class VectorQuantizer(nn.Module):
    """
    Vector Quantization layer.
    Options:
      - num_embeddings: size of codebook
      - embedding_dim: dimension of each code vector
      - commitment_cost (beta): weight for commitment loss
      - use_ema: if True use EMA updates (VQ-VAE-2 style); else use standard (sg) updates in loss
    Returns:
      quantized output, vq_loss, perplexity, encodings (one-hot index tensor)
    """
    def __init__(self, num_embeddings, embedding_dim, commitment_cost=0.25, use_ema=False, ema_decay=0.99, eps=1e-5):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        self.use_ema = use_ema

        # codebook
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        nn.init.uniform_(self.embedding.weight, -1.0 / num_embeddings, 1.0 / num_embeddings)

        if use_ema:
            # EMA buffers
            self.register_buffer('ema_cluster_size', torch.zeros(num_embeddings))
            self.register_buffer('ema_w', self.embedding.weight.data.clone())
            self.ema_decay = ema_decay
            self.eps = eps

    def forward(self, inputs):
        """
        inputs: (B, D, H, W) where D == embedding_dim
        returns:
          quantized: (B, D, H, W)
          loss: vq loss (commit + embedding update depending on mode)
          perplexity: float
          encodings: (B, H, W) indices
        """
        # move channel dim to last for distance computation
        # flatten: (B*H*W, D)
        input_shape = inputs.shape
        assert input_shape[1] == self.embedding_dim, "Input channels must match embedding_dim"
        flat_input = inputs.permute(0, 2, 3, 1).contiguous()  # (B, H, W, D)
        flat_input = flat_input.view(-1, self.embedding_dim)   # (N, D) where N=B*H*W

        # compute distances
        # ||x - e||^2 = ||x||^2 + ||e||^2 - 2 x.e
        embedding_weight = self.embedding.weight  # (K, D)
        distances = (
            torch.sum(flat_input**2, dim=1, keepdim=True)
            + torch.sum(embedding_weight**2, dim=1)
            - 2.0 * torch.matmul(flat_input, embedding_weight.t())
        )  # (N, K)

        # encoding indices
        encoding_indices = torch.argmin(distances, dim=1)  # (N,)
        encodings = F.one_hot(encoding_indices, self.num_embeddings).type(flat_input.dtype)  # (N, K)

        # quantized vectors
        quantized = torch.matmul(encodings, embedding_weight)  # (N, D)
        quantized = quantized.view(*input_shape[0:2], *input_shape[2:]).permute(0, 1, 2, 3)  # intermediate but we'll re-format below
        # Rebuild to (B, D, H, W)
        quantized = quantized.view(input_shape[0], input_shape[2], input_shape[3], self.embedding_dim).permute(0, 3, 1, 2).contiguous()

        # if self.use_ema:
        #     # EMA updates (only during training)
        #     if self.training:
        #         # Update cluster size
        #         dw = torch.sum(encodings, dim=0)  # (K,)
        #         self.ema_cluster_size = self.ema_cluster_size * self.ema_decay + (1 - self.ema_decay) * dw
        #
        #         # Update weights
        #         dw_weight = torch.matmul(encodings.t(), flat_input)  # (K, D)
        #         self.ema_w = self.ema_w * self.ema_decay + (1 - self.ema_decay) * dw_weight
        #
        #         # Laplace smoothing
        #         n = torch.sum(self.ema_cluster_size)
        #         cluster_size = ((self.ema_cluster_size + self.eps) / (n + self.num_embeddings * self.eps)) * n
        #
        #         # normalize ema_w to get new embeddings
        #         normalized_weight = self.ema_w / cluster_size.unsqueeze(1)
        #         self.embedding.weight.data.copy_(normalized_weight)
        #
        #     # commitment loss (L2 between encoder output and quantized)
        #     e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        #     loss = self.commitment_cost * e_latent_loss
        # else:
        # standard VQ: embedding loss + commitment loss
        # embedding loss: move embedding towards encoder output (stop gradient on input)
        embedding_loss = F.mse_loss(quantized, inputs.detach())
        commitment_loss = F.mse_loss(quantized.detach(), inputs)
        loss = embedding_loss + self.commitment_cost * commitment_loss

        # Straight-through estimator: pass gradients from quantized -> encoder inputs
        quantized = inputs + (quantized - inputs).detach()

        # perplexity
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        # reshape encoding indices to (B, H, W)
        encoding_indices = encoding_indices.view(input_shape[0], input_shape[2], input_shape[3])

        return quantized, loss, embedding_loss, commitment_loss, perplexity, encoding_indices


# -------------------------
# Decoder
# -------------------------
class Decoder(nn.Module):
    """
    Mirror of Encoder using transpose conv (or upsample + conv). Configuration mirrors Encoder.
    """
    def __init__(self,
                 input_channels,
                 out_channels=1,
                 hidden_channels=128,
                 n_up=4,
                 kernel_size=4,
                 use_batchnorm=False,
                 activation=nn.ReLU,
                 num_residual_blocks=0):
        super().__init__()

        # we'll start from hidden_channels * something similar to encoder's final channels
        in_ch = input_channels
        self.initial = ConvBlock(in_ch, in_ch, kernel=3, stride=1, padding=1,
                                 use_batchnorm=use_batchnorm, activation=activation)

        ups = []
        for i in range(n_up):
            out_ch = max(in_ch // 2, hidden_channels) if i < n_up - 1 else hidden_channels // 2
            # use ConvTranspose2d to upsample by 2 (kernel_size=kernel_size, stride=2, padding=kernel//2, output_padding=kernel%2)
            ups.append(nn.Sequential(
                nn.ConvTranspose2d(in_ch, out_ch, kernel_size=kernel_size, stride=2, padding=(kernel_size // 2 - 1),
                                   output_padding=kernel_size % 2),
                nn.BatchNorm2d(out_ch) if use_batchnorm else nn.Identity(),
                activation(inplace=True)
            ))
            in_ch = out_ch
        self.ups = nn.Sequential(*ups)

        # final conv to reconstruct to out_channels
        self.final = nn.Sequential(
            nn.Conv2d(in_ch, out_channels, kernel_size=3, stride=1, padding=1),
            nn.Tanh()  # often used if input normalized to [-1,1]; swap to Identity if using z-score / other norm
        )

        # optional residuals
        if num_residual_blocks > 0:
            res_blocks = []
            for _ in range(num_residual_blocks):
                res_blocks.append(nn.Sequential(
                    ConvBlock(in_ch, in_ch, kernel=3, stride=1, padding=1,
                              use_batchnorm=use_batchnorm, activation=activation),
                    nn.Conv2d(in_ch, in_ch, kernel_size=1)
                ))
            self.res_blocks = nn.ModuleList(res_blocks)
        else:
            self.res_blocks = None

    def forward(self, x):
        x = self.initial(x)
        x = self.ups(x)
        if self.res_blocks is not None:
            for r in self.res_blocks:
                res = r(x)
                x = x + res
        x = self.final(x)
        return x


# -------------------------
# Overall VQ-VAE
# -------------------------
class VQVAE(nn.Module):
    """
    VQ-VAE end-to-end model. Exposes forward(x) -> recon, total_loss, logs dict
    """
    def __init__(self,
                 image_channels=1,
                 hidden_channels=128,
                 embedding_dim=64,
                 num_embeddings=512,
                 n_down=4,
                 commitment_cost=0.25,
                 use_ema=False,
                 ema_decay=0.99,
                 use_batchnorm=False,
                 activation=nn.ReLU,
                 kernel_size=4,
                 num_residual_blocks=0,
                 recon_loss_weight=1.0):
        super().__init__()
        self.encoder = Encoder(in_channels=image_channels,
                               hidden_channels=hidden_channels,
                               n_down=n_down,
                               kernel_size=kernel_size,
                               use_batchnorm=use_batchnorm,
                               activation=activation,
                               num_residual_blocks=num_residual_blocks)
        # ensure embedding dim matches encoder channels (we project to embedding_dim)
        # compute encoder final channel output (mirror of Encoder)
        # compute encoder final channel output: MUST match the output of the Encoder's self.downs
        in_ch = hidden_channels
        for i in range(n_down):
            final_channels = min(hidden_channels * (2 if i > 0 else 1), 4 * hidden_channels)
            in_ch = final_channels  # in_ch tracks the output of the previous layer, which is the input to the current one
        # The final `final_channels` is the correct output channel count
        self.pre_vq_conv = nn.Conv2d(final_channels, embedding_dim, kernel_size=1)

        self.vq = VectorQuantizer(num_embeddings=num_embeddings,
                                  embedding_dim=embedding_dim,
                                  commitment_cost=commitment_cost,
                                  use_ema=use_ema,
                                  ema_decay=ema_decay)

        self.post_vq_conv = nn.Conv2d(embedding_dim, final_channels, kernel_size=1)

        self.decoder = Decoder(final_channels,
                               out_channels=image_channels,
                               hidden_channels=hidden_channels,
                               n_up=n_down,
                               kernel_size=kernel_size,
                               use_batchnorm=use_batchnorm,
                               activation=activation,
                               num_residual_blocks=num_residual_blocks)

        self.recon_loss_weight = recon_loss_weight

        self.embedding_dim = embedding_dim

    def forward(self, x):
        # x: (B, C, H, W)
        z_e = self.encoder(x)                     # (B, C_e, H_e, W_e)
        z_e = self.pre_vq_conv(z_e)               # (B, D, H_e, W_e) where D = embedding_dim
        quantized, vq_loss, embed_loss, commit_loss, perplexity, encoding_indices = self.vq(z_e)
        z_q = self.post_vq_conv(quantized)        # (B, C_e, H_e, W_e)
        x_recon = self.decoder(z_q)               # (B, C, H, W)

        # reconstruction loss
        recon_loss = F.mse_loss(x_recon, x) * self.recon_loss_weight

        # total loss
        total_loss = recon_loss + vq_loss

        # Commitment loss
        # Image

        logs = {
            "commitment_loss": commit_loss.detach(),
            "embedding_loss": embed_loss.detach(),
            "recon_loss": recon_loss.detach(),
            "vq_loss": vq_loss.detach(),
            "total_loss": total_loss.detach(),
            "perplexity": perplexity.detach() if isinstance(perplexity, torch.Tensor) else torch.tensor(perplexity)
        }
        return x_recon, total_loss, logs, encoding_indices
