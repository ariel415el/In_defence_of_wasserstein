
# In Defence of Wasserstein

This code repository contains the code base for training discreteWGANs, for direct patch SWD optimizationd and for OTmeans
that appear in our paper submission for ICLR 2024

# 1. General usage
## 1.1 Training a discreteWGAN
the `train.py` combined with the argument `--loss_function WGANLoss` trains a WGAN. adding `--z_prior const=64` make it have a discrete latent space

Fore example here is the python command we used to train discreteWGAN with a FC discriminator for Figure 2 in the paper
```
python3 train.py --data_path <data-path>  --z_prior const=64 --gen_arch FC --disc_arch FC-nf=1024 --lrD 0.001 --loss_function WGANLoss --gp_weight 10 --G_step_every 5 --train_name my_discreteWGAN-FC
```
Debug images will be written into 'outputs/train_results/my_discreteWGAN-FC'

## 1.2 Direct patch SWD optimization
We implemented the direct optimiztatio under the same script. The first thing is to change the loss function with the argument
`--loss_function MiniBatchLoss-dist=w1` for image level W1 or `--loss_function MiniBatchPatchLoss-dist=swd-p=16-s=1` for patch level SWD
Since this repository is based on a GAN training codebase and sice there is no discriminator needed for the direct optimization we have to tell
the train code not to train the non-existing discriminator using `--D_step_every -1`. 

Here is the command we used for the direct patchSWD optimization in Figure 3 of the paper
```
python3 train.py --data_path <data-path>  --z_prior const=64 --gen_arch Pixels --lrG 0.001 --loss_function MiniBatchPatchLoss-dist=swd-p=16-s=1 --D_step_every -1 --train_name my_DirectSWD
```
Debug images will be written into 'outputs/train_results/my_DirectSWD'

For both WGAN and direct Patch SWD you have a loook at all the possible arguments for the **train.py** script [here](utils/train_utils.py).

## 1.3 runing OTMeans
We implemented OTMeans in a [separate script](other_scripts/ot_means.py)
The usage is similar to the **train.py** script
Here is the command we used for generating the centroids in figure 2 in the paper:

```
python3 `other_scripts/ot_means.py` --data_path <data-path>  --k 64 --train_name my_OTmeans
```
Debug images will be written into 'outputs/my_OTmeans'

# 2. Reproducing the paper's figures
All the experiments below are performed on the three datasets described in the paper with the following dataset specific arguments
```
squares: --gray_scale
MNIST: --limi_data 10000 --gray_scale
FFHQ: --limi_data 10000 --center_crop 80
```

## 2.1 Figure1, Figure 4 and Appendix A figures:
Running
```
python3  other_scripts/batch_size_effect.py --data_path <FFHQ-path> 
python3  other_scripts/batch_size_effect.py --data_path <MNIST-path>
python3  other_scripts/batch_size_effect.py --data_path <squares-path>
```
will create minibatches of real, repeated-means and OT-centroids in sizes [10,100,500,1000] and compare them to a batch of other images
the outputs will apppear at 'outputs/batch_size_effect/<dataset-name>'


## 2.2 Figure2: DiscreteWGAN with FC discriminator vs OTmeans

```
python3 train.py --data_path <data-path>  --z_prior const=64 --gen_arch FC --disc_arch FC-nf=1024 --lrD 0.001 --loss_function WGANLoss --gp_weight 10 --G_step_every 5
python3 `other_scripts/ot_means.py` --data_path <data-path>  --k 64 
```

## 2.3 Figure3: DiscreteWGAN with CNN+GAP discriminator vs direc Patch SWD
```
python3 train.py --data_path <data-path>  --z_prior const=64 --gen_arch FC --disc_arch PatchGAN --lrD 0.001 --loss_function WGANLoss --gp_weight 10 --G_step_every 5
python3 train.py --data_path <data-path>  --z_prior const=64 --gen_arch Pixels --lrG 0.001 --loss_function MiniBatchPatchLoss-dist=swd-p=16-s=1 --D_step_every -1 
```


## 2.3 Figure5: CNN+GAP discriminator vs CNN+FC discriminator
```
python3 train.py --data_path <data-path>  --z_prior const=64 --gen_arch FC --disc_arch PatchGAN-nf=256-GAP=True --lrD 0.001 --loss_function WGANLoss --gp_weight 10 --G_step_every 5
python3 train.py --data_path <data-path>  --z_prior const=64 --gen_arch FC --disc_arch PatchGAN-nf=256-GAP=FALSE --lrD 0.001 --loss_function WGANLoss --gp_weight 10 --G_step_every 5
```

## 2.4 Figure6: Generating sharp images with direct patch SWD optimization
```
python3 train.py --data_path <data-path> --z_prior const=64 --gen_arch Pixels --lrG 0.01 --D_step_every -1 --n_iterations 2000  --loss_function MiniBatchMSPatchLoss-dists='["w1","swd"]'-ps='[64, 8]'-ss='[1,4]'-intervals='[1000]'
```

In order to find the data nearest neighbors for generated images needed for Figure 11 of the paper please run
the following command and specify the directory with the outputs of the above command
```
python3 other_scripts/find_nearest_neighbors.py <path to trained model>
```
the NN plot will appear in a subfolder named 'test_outputs' int the trained model directory specified.

# Data
### squares dataset: run 
```
python3 other_scripts/create_squares_dataset.py 
```
will create the datsaet 

### FFHQ
Download 128x128 thumbnails from https://github.com/NVlabs/ffhq-dataset

### MNIST
Download MNIST from http://yann.lecun.com/exdb/mnist/
We used [store_mnist_as_png.py](store_mnist_as_png.py) to store the dataset as pngs for the training scripts to load

# Credits
Codebase is based on https://github.com/odegeasslbc/FastGAN-pytorch
