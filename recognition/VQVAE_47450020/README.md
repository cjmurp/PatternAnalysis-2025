# Generative Modelling of HipMRI Prostate Cancer Slice Images Using VQ-VAE

### Author: Corey Murphy, University of Queensland - s4745002 

## 1. Overview
This project implements a generative model based on the Vector Quantised Variational Encoder (VQ-VAE) 
architecture to synthesise realistic 2D prostate MRI slices from the HipMRI study on prostate cancer dataset. 
The goal is to produce high-quality, "reasonably clear" generated images with a Structural Similarity 
Index Measure (SSIM) above 0.6.

The VQ-VAE learns a discrete latent space representation of the MRI data, allowing it to capture complex spatial and
structural patterns present in the prostate MRIs. This discrete latent space can then be used to generate
realistic prostate MRI images. 
Typical VAEs use a continuous latent space, which often fails to preserve fine image details, resulting in 
blurry reconstructions. This limitation makes standard VAEs less suitable for medical imaging applications where
precision is essential.

This project will demonstrate how the VQ-VAE architecture overcomes these challenges by using vector quantisation
to generate a discrete latent space to produce sharper, more realistic medical images.

## 2. Background and Motivation

Variational Autoencoders (VAEs) consist of two core components, an encoder and a decoder. The encoder maps
an input image to a continuous latent representation by predicting parameters of a probability distribution. This is 
typically a Gaussian distribution, where the model aims to learn the mean and variance. VAEs are stable and learn this 
latent space, but they tend to produce blurred images due to their continuous latent sampling and Gaussian distribution.

![alt text](./Resources/VAE_image.png "Title")

Vector Quantised Variational Encoders (VQ-VAEs) introduce a discrete latent space through the use of a codebook 
(embedding dictionary) of learned latent vectors. The encoder outputs are quantised by mapping the encoder output
to the "closest" vector in the codebook. The encoder, using a convolutional neural network (CNN), reduces the input
image into a lower-dimensional latent representation that captures key spatial and structural features. This can be 
quantised by mapping it to the closest entry in the codebook based on a distance metric. This encoder output
is essentially replaced with the nearest codebook vector. This process removes the need to sample a continuous space by 
discretising it to distinct vectors, which correspond to meaningful patterns in the data.
The decoder then reconstructs the image from these quantised embedding vectors, producing sharper images. The decoder
is another CNN that reconstructs the image from its compressed representation.

Training uses three terms:
- Reconstruction Loss: Difference between input and output
- Codebook Loss: Encourages the embeddings to adapt to the encoder outputs
- Commitment Loss: Prevents the encoder from moving too far away from the set of embeddings

Quantisation makes the process non-differentiable, meaning gradient flow for backpropagation will not work. 

A straight-through estimator, which connects the encoder and the decoder directly, is used to get around this. This
passes the gradients from the decoder directly back to the encoder. This is represented on the diagram below as a
forward pass marked as the red arrow. This allows the reconstruction loss to be used to tune the model.

![alt text](./Resources/VQ_VAE_image.jpg "Title")

Since there is this pass-through, there is now no connection to the embedding vectors to tune them during backpropagation.
This is where the Codebook loss and commitment loss are used. The codebook loss is used to bring the embedding vectors
closer to the encoder output being mapped to them. Each codebook vector can be treated as a centroid to the various
encoded points that are assigned to them. This would slowly update the embedded vectors to be close to the encoder output.
Commitment loss is also used to make sure the encoder commits to an embedding and prevents the encoder outputs from
fluctuating between different code vectors, bringing stability to the training. The codebook vector is treated as 
"stationary" and the encoded points are brought closer to that point. 

## 3. Dataset and Pre-Processing
The HipMRI Prostate 2d image slice dataset was used for this project. The data set consists of greyscale images that are stored in nifti format. 
A custom DataSet class was created to load these images and convert them to standard images with values ranging from 0 to 255. This custom
DataSet class can be seen in predict.py. The custom class is set up so that these loaded images are processed with a specified transform. 
By default, this transform just converts the image to a Tensor for later processing. 

For better performance, a specific set of transforms was set. 

``` python
transform = transforms.Compose([
        transforms.Resize((256, 128)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
```
The specified transform ensures all images are a consistent size to prevent sizing issues in the model. It also ensures that the image is grayscale,
ensuring there is only one output channel. This is again to keep dimensions consistent for the model. It is then converted to a Tensor and then immediately
normalised. It is being normalised to the range of -1 to 1. This improves stability and works very well with Tanh activation function. 

### Datasplit
The overall data has been split into distinct groups to properly train and evaluate the model. The split is:

**train:** 11460 (90.5%)  
**test:** 540 (4.3%)  
**validate:** 660 (5.2%)  
**Total:** 12660  

This split is the standard split that is present in the keras_slices_data for the HipMRI Prostate study. This split is desirable. A large training set
is needed to ensure the model learns the appropriate features and does not over overfit to the small data set. The percentage of 90 percent fits
this purpose. A validate set was used while training to ensure that the model was not overfitting to the training set. This means SSIM and loss 
can be plotted for the training and validate set for each epoch to appropriately monitor progress. Finally, the test set needs to be seperate again to
ensure the model has not overfitted to the training set and can output adequete results for unseen data. Both the validate and test set do not need to
be as big as the training set, leaving 10 percent of the data between them. This split was inputted into three seperate dataloaders to be used by the model.

## 4. Model Architecture
The VQ-VAE model consists of three main components:
**Encoder**, **Vector Quantizer (Codebook)**, and **Decoder**.  
Together, they learn a discrete latent representation of input data by mapping continuous encoder outputs to the nearest entries in a learned embedding space. 

### Encoder
The encoder compresses the input image into a lower dimensional latent representation. The input image passes through several convolution layers that downsample
the image. Each layer downsamples but it increases the feature depth. The encoder outputs a continuous latent space. The layers used consist of initial convolutions
and downsampling convolutions. The initial layers ensure the channel dimension match the necessary number of hidden channels. The downsampling channels are then used to
reduce the image size. An input variable n_down defines the number of downsampling layers. These layers can be visualised in the table below.

| Layer | Type | Output Shape | Notes |
|--------|------|---------------|-------|
| 1 | Conv2D (stride=1, kernal=3) + ReLU | H × W | Initial Convolution, increase channels to: in_channels -> hidden_channels // 2 |
| 2 | Conv2D (stride=1, kernal=3) + ReLU | H × W | Increase channels to: hidden_channels // 2 -> hidden_channels |
| 3 | Conv2D (stride=2, kernal=4) + ReLU | ↓H/2 × ↓W/2 | Downsample, reduce H and W by two |
| ... | Conv2D (stride=2, kernal=4) + ReLU | ↓H/2 × ↓W/2 | Further downsample. Input variable n_down defines how many downsamples  |
| Output | None | D × H' × W' | Embedding dimension |

The encoder learns to capture semantic features (edges, textures, tissue structure, etc.) in the compressed latent space.

### Quantizer
The vector quantizer replaces the continuous latent vectors with discrete vector mappings. The quantizer is defined as a VectorQuantizer class in modules.py.
The use of a codebook to represent the discrete latent vector space is what makes VQ-VAE different from a typical VAE. Within the class, the number of embeddings
and the embedding dimensions are both defined and used to build the codebook. Torch.nn provides a simple way to initalise this:
```
# codebook
self.embedding = nn.Embedding(num_embeddings, embedding_dim)
```
The forward function then calculates how the encoder outputs map to this codebook. For each continuous latent vector, the nearest embedding is found using Euclidean
distance. This is done by calculating all distances and selecting the shortest. This is represented in the forward function of the VectorQuantizer class:
```
encoding_indices = torch.argmin(distances, dim=1)
```
The selected embedding replaces the encoder output, producing the quantised latent map. To ensure the embeddings accurately represent the encoder outputs, an embedding loss is used. This will try to move the codebook vectors towards the encoder outputs. The embedding loss is defined below, where z_e(x) represents the quantized vectors and the e_k is the inputs. The sg means stop gradient. This ensures back propogation does npt flow back through the outlined variable set. For the below equation the model will update itself based on the distance the codebook vectors are from the encoder outputs. Effectively bringing the defined codebook entries closer to the features that
are outputted by the encoder:

   $$\mathcal{L}_{\text{embedding}} = \| z_e(x) - \text{sg}[e_k] \|^2$$
   
This is represented in the code as:
```
embedding_loss = F.mse_loss(quantized, inputs.detach())
```
Additionally, a commitment loss was used. The Commitment loss ensures that the encoder outputs stay close to their chosen embeddings. This prevents any oscillatory behaviour between the mapped codebook vector and the encoder outputs. Similarly the encoder vectors now have no gradient flowing through them. This effectively updates the encoder outputs (e_k) such that it minimises this distance.

   $$\mathcal{L}_{\text{commit}} = \| \text{sg}[z_e(x)] - e_k \|^2$$

This is represented in the code as:
```
commitment_loss = F.mse_loss(quantized.detach(), inputs)
```
The overall quantizer error can be calculated. The commitment loss is scaled down by a variable:

$$\mathcal{L}_{\text{commit}} = \mathcal{L}_{\text{embedding}} + \beta * \mathcal{L}_{\text{commitment}}$$

In code:
```
loss = embedding_loss + self.commitment_cost * commitment_loss
```

### Decoder
The decoder takes the output quantised latent space and reconstructs the image. Taking the latent map as input, it uses transposed convolutions or
upsampling layers to progressively reconstruct the spatial resolution. This outputs the reconstructed image

| Layer | Type | Output Shape | Notes |
|--------|------|---------------|-------|
| 1 | Conv (stride=1, kernal=3) | D × H' × W' | Initial Convolution |
| 2 | ConvTranspose2D (stride=2, kernal=4) | ↑2*H × ↑2*W | Upsampling, increase H and W by two |
| ... | ConvTranspose2D (stride=2, kernal=4) | ↑2*H × ↑2*W | Further upsampling, Input variable n_down defines how many upsamples |
| Pre-Output | ConvTranspose2D (stride=1, kernal=3) | ↑H × ↑W | Final upsample to ensure correct number of channels |
| Output | Activation (Tanh) | ↑H × ↑W | Final reconstruction using activation function |


### Overall Model
The VQ-VAE model was adapted from **REFERENCE**

The overall model uses the above components with a convolution layer on either side of the quantizer to ensure channels and dimensions match:
```
    def forward(self, x):
        # x: (B, C, H, W)
        z_e = self.encoder(x)                     # (B, C_e, H_e, W_e)
        z_e = self.pre_vq_conv(z_e)               # (B, D, H_e, W_e) where D = embedding_dim
        quantized, vq_loss, embed_loss, commit_loss, perplexity, encoding_indices = self.vq(z_e)
        z_q = self.post_vq_conv(quantized)        # (B, C_e, H_e, W_e)
        x_recon = self.decoder(z_q)               # (B, C, H, W)
```
Where:
```
self.pre_vq_conv = nn.Conv2d(final_channels, embedding_dim, kernel_size=1)
self.post_vq_conv = nn.Conv2d(embedding_dim, final_channels, kernel_size=1)
```

## 5. Training
The train.py file contains the code to train the model. This uses the VQVAE class in modules.py and the custom dataset from dataset.py.

### Key Variables
The VQ-VAE model was trained with the key variables:

| Variable | Description | Value / Options |
|----------|-------------|----------------|
| `num_epochs` | Total number of training epochs | 20 |
| `batch_size` | Number of images per batch | 32 |
| `learning_rate` | Initial learning rate for optimizer | 1e-3 |
| `hidden_channels` | Base number of feature channels in encoder/decoder | 128 |
| `embedding_dim` | Dimension of the codebook embeddings | 128 |
| `num_embeddings` | Number of embeddings in the codebook | 512 |
| `beta` | Weight for the commitment loss in VQ-VAE | 0.25 |
| `n_down` | Number of down sampling layers | 4 |

### Training Process
After data preperation described above, the images were passed through to the model in batches of 32. This size ended up being an ideal size. Anything smaller was quicker but learned less/slower, which needed more epochs to get to the same result. The overall loss was then calculated. The reconstruction loss used MSE between the inputted image and the outputted reconstructed image. The total loss was the sum of this and the quantisation loss defined in section 4. An Adam optimiser was used with a learning rate of 0.001. The Adam Optimiser provided adaptive learning rates, providing better training of the model and more effectively finding lower and lower loss. 

### Results
Each epoch, the loss and the SSIM scores were calculated and plotted after training was done. Additionaly the input image and the reconstructed image for the epochs were also visualised to track progress.

After 1 epoch:

![alt text](./Resources/train/epoch_1/recon_imag_0.png "Title")
![alt text](./Resources/train/epoch_1/recon_imag_1.png "Title")
![alt text](./Resources/train/epoch_1/recon_imag_2.png "Title")

After 5 epoch:

![alt text](./Resources/train/epoch_5/recon_imag_0.png "Title")
![alt text](./Resources/train/epoch_5/recon_imag_1.png "Title")
![alt text](./Resources/train/epoch_5/recon_imag_2.png "Title")
After 10 epoch:

![alt text](./Resources/train/epoch_10/recon_imag_0.png "Title")
![alt text](./Resources/train/epoch_10/recon_imag_1.png "Title")
![alt text](./Resources/train/epoch_10/recon_imag_2.png "Title")

After 20 epoch:

![alt text](./Resources/train/epoch_20/recon_imag_0.png "Title")
![alt text](./Resources/train/epoch_20/recon_imag_1.png "Title")
![alt text](./Resources/train/epoch_20/recon_imag_2.png "Title")

The SSIM Scores:
![alt text](./Resources/train/ssim_epoch_losses.png "Title")

The SSIM values quickly passed this minimum of 0.6 around 3 epochs. With it steadily increasing to above 0.7. This quick initial learning suggest that the VQ-VAE learns key features very quickly but takes longer to fine tune. The validation and training outputs follow each other closely, meaning the model is not overfitting to the training set. 

The loss plot can also be plotted:
![alt text](./Resources/train/vqvae_epoch_losses.png "Title")

The loss values are quite low, indicating that the model has performed well in reconstructing the images. 
## 6. Testing and Results

To evaluate the model properly, code was created in predict.py. The SSIM scores for all of the images can be plotted to visualise the variance:
![alt text](./Resources/test/test_eval/test_ssim_curve.png "Title")
With 

**Max SSIM:** 0.8261  
**Min SSIM:** 0.6458  
**Average SSIM:** 0.7422  

All values fit above the minimum SSIM value of 0.6, providing "Reasonably clear" images. There does seem to be some variance in the outputs suggesting that the model does not perform as consistently as desired, possibly due to the model performing well with input images that are similar to what it was trained on and struggling with the images that differ the most to the training set. The model still generalises quite well to keep the minimum scores above the threshold though.

Some Visualisations of the test set can be seen:

![alt text](./Resources/test/test_eval/reconstruction_0.png "Title")
![alt text](./Resources/test/test_eval/reconstruction_1.png "Title")
![alt text](./Resources/test/test_eval/reconstruction_2.png "Title")
![alt text](./Resources/test/test_eval/reconstruction_3.png "Title")

Visualisation of the codebook embedding:
![alt text](./Resources/test/codebook_vis/codebook_map_0.png "Title")

## 7. How to Run

### Dependencies
| Dependency    | Version |
|---------------|---------|
| Python        | 3.9.23  |
| PyTorch       | 2.8.0   |
| Torchvision   | 0.23.0  |
| NumPy         |  1.26.3 |
| Matplotlib    | 3.9.4   |
|  tqdm         |  4.67.1 |
| torchmetrics  |  1.8.2  |
|  nibabel      |  5.3.2  |


These can be installed by:
```
pip install pytorch torchvision numpy matplotlib tqdm torchmetrics nibabel
```
### Running 
The results found can be recreated by running the attached files. Initially the model needs to be trained by running
```
python train.py
```
This will save checkpoint for each epoch, saving the respective model, and output visualisations of the training. Image reconstructions for each epoch is also visualised and saved.
The model can then be evaluated by running:
```
python predict.py
```
This will plot the SSIM and loss scores and visualise some reconstructions. This will also look at the codebook embeddings and visualise them. All images are saved.

## References
- - - - -
