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
is needed to ensure the model learns the appropriate features and does not over generalise to the small data set. The percentage of 90 percent fits
this purpose. A validate set was used while training to ensure that the model was not generalising to the training set. This means SSIM and loss 
can be plotted for the training and validate set for each epoch to appropriately monitor progress. Finally, the test set needs to be seperate again to
ensure the model has not generalised to the training set and can output adequete results for unseen data. Both the validate and test set do not need to
be as big as the training set, leaving 10 percent of the data between them. This split was inputted into three seperate dataloaders to be used by the model.

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
