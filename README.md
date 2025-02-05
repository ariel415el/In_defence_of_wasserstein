# In Defense of Wasserstein: Understanding WGANs Through Discrete Generators  

- [![ICLR 2025](https://img.shields.io/badge/ICLR-2025-blue)](https://iclr.cc/virtual/2025/poster/30814) Link to our paper on ICLR2025    
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ariel415el/DiscreteGANs/blob/ICLR2025/Lessons_from_discrete_GANs.ipynb) Reproduces the paper experiments.
  
## 📜 Abstract  
Since WGANs were first introduced, there has been considerable debate about whether their success in generating realistic images can be attributed to minimizing the Wasserstein distance between the generated and training distributions. In this paper, we present theoretical and experimental results showing that successful WGANs **do** minimize the Wasserstein distance, but the specific form of the distance minimized depends heavily on the discriminator architecture and its inductive biases.  

Specifically, we show that when the discriminator is convolutional, WGANs minimize the Wasserstein distance **between patches** in generated and training images rather than the distance between full images. Our results are obtained using **discrete generators**, where the Wasserstein distance between distributions can be computed exactly and analytically characterized. We present experiments demonstrating that discrete GANs can generate realistic images (comparable in quality to continuous counterparts) while minimizing the Wasserstein distance between patches rather than full images.  

# General usage
## 1. Training a discreteWGAN
the `train.py` combined with the argument `--loss_function WGANLoss` trains a WGAN. The `--z_prior` arguments control the 
argumnet type using `--z_prior const=N` make it have a discrete latent space of size `N`

Fore example here is the python command we used to train discreteWGAN with a FC discriminator for Figure 5 in the paper
```
python3 train.py --data_path <data-path>  --z_prior const=64 --gen_arch FC --disc_arch FC-nf=1024 --lrD 0.001 --loss_function WGANLoss --gp_weight 10 --G_step_every 5 --train_name my_discreteWGAN-FC
```
Debug images will be written into 'outputs/train_results/my_discreteWGAN-FC'

## 2. Direct patch SWD optimization
We implemented the direct optimiztatio under the same script. The first thing is to change the loss function with the argument
`--loss_function MiniBatchLoss-dist=w1` for image level W1 or `--loss_function MiniBatchPatchLoss-dist=swd-p=16-s=1` for patch level SWD.
Since there is no discriminator needed for the direct optimization we have to tell
the train code not to train the non-existing discriminator using `--D_step_every -1`. 

Here is the command we used for the direct patchSWD optimization in Figure 7 of the paper
```
python3 train.py --data_path <data-path>  --z_prior const=64 --gen_arch FC --lrG 0.001 --loss_function MiniBatchPatchLoss-dist=swd-p=16-s=1 --D_step_every -1 --train_name my_DirectSWD
```
Debug images will be written into 'outputs/train_results/my_DirectSWD'

For both WGAN and direct Patch SWD you have a loook at all the possible arguments for the **train.py** script [here](utils/train_utils.py).

## 3. runing OTMeans
We implemented OTMeans in a [separate script](scripts/ot_means.py)
The usage is similar to the **train.py** script
Here is the command we used for generating the centroids in figure 2 in the paper:

```
python3 `other_scripts/ot_means.py` --data_path <data-path>  --k 64 --train_name my_OTmeans
```
Debug images will be written into 'outputs/my_OTmeans'

# Reproducing the paper's figures
All experiments of Mnist and Squares datasets were done with the --gray_scale argument
All experiment on FFHQ were done with --center crop 90

## Figure 2: SOTA discrete GAN 
We trained this model on 128x128 images but in order to crop a more focused images around the face not upscale from 90 to 128 
for this experiment we used the 1024x1024 version of FFHQ and center cropped a 720x720 images which we resized to 128 
```
python3 train.py --data_path FFHQ_HQ_cropped  --limit_data 70000 --project_name trainFastGAN --log_freq 5000 --f_bs 16 --r_bs 16 --z_dim 128 --im_size 128 --z_prior const=70000 --gen_arch FastGAN --lrD 0.0002 --lrG 0.0001 --loss_function WGANLoss --gp_weight 10 --train_name ffhq_hq_const=70000_I-128_Z-128_Reg-GP_G-OldFastGAN_D-OldFastGAN  --disc_arch OldFastGAN --wandb --full_batch_metrics
```

## Figure 5: DiscreteWGAN with FC discriminator vs OTmeans

```
python3 train.py --data_path <data-path>  --z_prior const=64 --gen_arch FC --disc_arch FC-nf=1024 --lrD 0.001 --loss_function WGANLoss --gp_weight 10 --G_step_every 5
python3 scripts/ot_means.py --data_path <data-path>  --k 64 
```

## Figure 6: DiscreteWGAN copies data when M=N
```
python3 train.py --data_path <data-path> --limit_data 1000  --z_prior const=1000 --gen_arch FC --disc_arch FC-nf=1024 --lrD 0.001 --loss_function WGANLoss --gp_weight 10 --G_step_every 5
```


## Figure 7: DiscreteWGAN: CNN+GAP discriminator vs CNN+FC
```
python3 train.py --data_path <data-path>  --z_prior const=64 --gen_arch FC --disc_arch CNN-GAP=True --lrD 0.001 --loss_function WGANLoss --gp_weight 10 --G_step_every 5
python3 train.py --data_path <data-path>  --z_prior const=64 --gen_arch FC --disc_arch CNN-GAP=False --lrD 0.001 --loss_function WGANLoss --gp_weight 10 --G_step_every 5
```

# Data
### squares dataset: run 
```
python3 scripts/create_squares_dataset.py 
```
will create the dataset 

### FFHQ
Download 128x128 thumbnails from https://github.com/NVlabs/ffhq-dataset

### MNIST
Download MNIST from http://yann.lecun.com/exdb/mnist/
We used [store_mnist_as_png.py](scripts/store_mnist_as_png.py) to store the dataset as pngs for the training scripts to load

# Credits
Codebase is based on https://github.com/odegeasslbc/FastGAN-pytorch
