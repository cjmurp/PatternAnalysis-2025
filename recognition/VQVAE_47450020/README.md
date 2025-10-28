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
Describe the dataset and how to access and load  
What pre-processing is used?  
Splitting training, validation and test  
## 4. Model Architecture
Describe VAQVAE  
Layers and breakdown  
Hyperparameters  

## 5. Training
Key parameters   
Training duration, epochs, checkpoints  
Training loss plot   
SSIM plot  
## 6. Testing and Results

## 7. How to Run
Install and set up environment  
Dataset setup  
training command  
testing command  
## Dependencies


## References
- - - - -
