from plot_train_results import plot


if __name__ == '__main__':
    out_root = 'outputs'
    named_dirs = {
        "FC-FC": f"{out_root}/GANs/images_I-64x64_G-FC_D-FC-nf=1024_L-WGANLoss_Z-64xconst=128_B-128-128_test",
        "Pixels-FC": f"{out_root}/GANs/images_I-64x64_G-Pixels-n=128_D-DCGAN-normalize=none_L-WGANLoss_Z-64xconst=128_B-128-128_test",
        "Pixels-DC": f"{out_root}/GANs/images_I-64x64_G-Pixels-n=128_D-FC-nf=1024_L-WGANLoss_Z-64xconst=128_B-128-128_test",

    }
    plot(named_dirs, f"{out_root}/Exp-arbitrary.png", plot_loss="separate", n=3)