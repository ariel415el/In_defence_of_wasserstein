

def get_generator(name, res, z_dim):
    if name  == "FastGAN":
        if res != 128:
            raise ValueError("FastGan only implemented for 128x128 images")
        from models.FastGAN import Generator, weights_init
        netG = Generator(z_dim, skip_connections=False)
        netG.apply(weights_init)

    elif name == 'DCGAN':
        if res != 64:
            raise ValueError("FastGan only implemented for 128x128 images")
        from models.DCGAN import Generator, weights_init
        netG = Generator(z_dim)
        netG.apply(weights_init)

    print("G params: ", sum(p.numel() for p in netG.parameters() if p.requires_grad))
    return netG


def get_discriminator(name, res, num_outputs):
    if name == "FastGAN":
        if res != 128:
            raise ValueError("FastGan only implemented for 128x128 images")
        from models.FastGAN import Discriminator, weights_init
        netD = Discriminator(num_outputs=num_outputs)
        netD.apply(weights_init)

    elif name == 'DCGAN':
        if res != 64:
            raise ValueError("FastGan only implemented for 64x64 images")
        from models.DCGAN import Discriminator, weights_init
        netD = Discriminator(num_outputs=num_outputs)
        netD.apply(weights_init)

    elif name =="BagNet-9":
        from models.BagNet import BagNet, Bottleneck
        netD = BagNet(Bottleneck, [3, 4, 6, 3], strides=[2, 2, 2, 1], kernel3=[1, 1, 0, 0], num_classes=1)

    print("D params: ", sum(p.numel() for p in netD.parameters() if p.requires_grad))
    return netD


def get_models(args, device):
    netG = get_generator(args.Generator_architecture, args.im_size, args.z_dim).to(device)
    netD = get_discriminator(args.Discriminator_architecture, args.im_size, num_outputs=1).to(device)

    return netG, netD