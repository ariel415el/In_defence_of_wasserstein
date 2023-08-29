import os
import sys
import torch
from matplotlib import pyplot as plt

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from losses.optimal_transport import MiniBatchLoss, MiniBatchPatchLoss
from scripts.EMD.utils import get_data, batch_to_image, read_grid_batch


if __name__ == '__main__':
    device = torch.device('cpu')
    b = 64
    im_size=64
    s = 5

    # data_path = '/cs/labs/yweiss/ariel1/data/square_data/black_S-10_O-1_S-1'
    # c=1
    # names_and_paths = [
    #     ("direct-w1", '../../outputs/TMP-Figure_Exp3/black_S-10_O-1_S-1_I-64x64_G-Pixels_D-DCGAN_GS_L-MiniBatchLoss-dist=w1_Z-64xconst=64_B-64-64_Exp3-Discrete-W1-black_S-10_O-1_S-1_08-16_T-10:22:15/images/3000.png'),
    #     ("direct-patch-w1", '../../outputs/TMP-Figure_Exp3/black_S-10_O-1_S-1_I-64x64_G-Pixels_D-DCGAN_GS_L-MiniBatchPatchLoss-dist=w1-p=16-s=8-n_samples=1024_Z-64xconst=64_B-64-64_Exp3-Discrete-W1_p=16-s=8-black_S-10_O-1_S-1_08-16_T-10:22:17/images/46000.png')
    # ]
    # data = get_data(data_path, im_size, c=c, center_crop=None, gray_scale=True, flatten=False, limit_data=None).to(device)

    # data_path = '/cs/labs/yweiss/ariel1/data/MNIST/MNIST/jpgs/training'
    # c=1
    # names_and_paths = [
    #     ("direct-w1", '../../outputs/garbage/training_I-64x64_G-Pixels_D-DCGAN_GS_L-MiniBatchLoss-dist=w1_Z-64xnormal_B-64-64_test/images/28000.png'),
    #     ("direct-patch-w1", '../../outputs/garbage/training_I-64x64_G-Pixels_D-DCGAN_GS_L-MiniBatchPatchLoss-dist=w1-p=16-s=8-n_samples=1024_Z-64xnormal_B-64-64_test/images/5000.png')
    # ]
    # data = get_data(data_path, im_size, c=c, center_crop=None, gray_scale=True, flatten=False, limit_data=10064).to(device)


    # data_path = '/cs/labs/yweiss/ariel1/data/FFHQ/FFHQ'
    # c=3
    # names_and_paths = [
    #     ("w1", 'outputs/TMP-Figure_test_GAN_patch_loss/FFHQ_I-64x64_G-DCGAN-normalize=in-nf=128_D-DCGAN-normalize=in-nf=128_CC-100_L-MiniBatchLoss_Z-64xnormal_B-64-64_Direct-W1-FFHQ_08-17_T-20:17:30/images/20000.png'),
    #     ("wgan", 'outputs/TMP-Figure_test_GAN_patch_loss/FFHQ_I-64x64_G-DCGAN-normalize=in-nf=256_D-DCGAN-normalize=in-nf=256_CC-100_L-SoftHingeLoss_Z-64xnormal_B-64-64_GAN-FFHQ_08-18_T-13:35:14/images/90000.png')
    # ]
    # data = get_data(data_path, im_size, c=c, center_crop=100, gray_scale=False, flatten=False, limit_data=10064).to(device)

    data_path = '/cs/labs/yweiss/ariel1/data/FFHQ/FFHQ_frontal_1000'
    c=1
    names_and_paths = [
        ("wgan-dc", 'outputs/GANs/FFHQ_frontal_1000_I-64x64_G-DCGAN-normalize=none_D-DCGAN-normalize=none_GS_CC-80_L-SoftHingeLoss_Z-64xnormal_B-64-64_test/images/51000.png'),
        ("gan-gap", 'outputs/GANs/FFHQ_frontal_1000_I-64x64_G-DCGAN-normalize=none_D-PatchGAN-normalize=none_GS_CC-80_L-SoftHingeLoss_Z-64xnormal_B-64-64_test/images/66000.png'),
        ("w1", 'outputs/GANs/FFHQ_frontal_1000_I-64x64_G-DCGAN-normalize=none_D-DCGAN_GS_CC-80_L-MiniBatchLoss_Z-64xnormal_B-64-64_test/images/7000.png')
    ]
    data = get_data(data_path, im_size, c=c, center_crop=80, gray_scale=True, flatten=False, limit_data=1000).to(device)



    dist = 'nn'
    nn_image = MiniBatchLoss(dist='swd')
    nn_patch = MiniBatchPatchLoss(dist='swd', p=8, s=4)

    # w1_image = MiniBatchLoss(dist='w1')
    # w1_patch = MiniBatchPatchLoss(dist='w1', p=16, s=8)

    real_batch = data[:b]
    data = data[b:]


    names_and_batchs = [("Real", real_batch)] + [(name,  read_grid_batch(path, im_size, c, flatten=False).to(device)) for name, path in names_and_paths]

    fig, ax = plt.subplots(nrows=1, ncols=len(names_and_batchs), figsize=(len(names_and_batchs)*s, s * 1.1))
    for i, (name, batch) in enumerate(names_and_batchs):
        ax[i].imshow(batch_to_image(batch, im_size, c))
        ax[i].axis('off')
        ax[i].set_title(f"{name}"
                        # f"\nW1: Image:{w1_image(batch, data).item():.3f}, Patch: {w1_patch(batch, data).item():.3f}" 
                        f"\nNN: Image:{nn_image(batch, data).item():.3f}, Patch: {nn_patch(batch, data).item():.3f}"
                        )

    plt.tight_layout()
    plt.savefig('plot.png')
