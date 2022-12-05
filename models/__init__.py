def get_generator(name, res, z_dim):
    if name == "FastGAN":
        if res != 128:
            raise ValueError("FastGan only implemented for 128x128 images")
        from models.FastGAN import Generator, weights_init
        netG = Generator(z_dim, skip_connections=False)
        netG.apply(weights_init)

    elif name == 'DCGAN':
        if res != 64:
            raise ValueError("DCGAN only implemented for 64x64 images")
        from models.DCGAN import Generator, weights_init
        netG = Generator(z_dim)
        # netG.apply(weights_init)

    elif name == 'FC':
        from models.FC import Generator
        netG = Generator(z_dim, output_dim=res)

    elif name == 'Resnet':
        from models.Resnet import ResNet_G
        netG = ResNet_G(z_dim, size=res)

    return netG


def get_discriminator(name, res):
    if name == "FastGAN":
        if res != 128:
            raise ValueError("FastGan only implemented for 128x128 images")
        from models.FastGAN import Discriminator, weights_init
        netD = Discriminator()
        netD.apply(weights_init)

    elif name == 'DCGAN':
        if res != 64:
            raise ValueError("DCGAN only implemented for 64x64 images")
        from models.DCGAN import Discriminator, weights_init
        netD = Discriminator()
        # netD.apply(weights_init)

    elif 'BagNet' in name:
        from models.BagNet import BagNet, Bottleneck
        kernel_dict = {"BagNet-9": [1, 1, 0, 0], "BagNet-17": [1, 1, 1, 0], "BagNet-33": [1, 1, 1, 1]}
        netD = BagNet(Bottleneck, kernel3=kernel_dict[name], num_classes=1)

    elif name == 'FC':
        from models.FC import Discriminator
        netD = Discriminator(in_dim=res)

    elif name == 'Resnet':
        from models.Resnet import ResNet_D
        netD = ResNet_D(size=res)

    return netD


def get_models(args, device):
    netG = get_generator(args.Generator_architecture, args.im_size, args.z_dim).to(device)
    netD = get_discriminator(args.Discriminator_architecture, args.im_size).to(device)

    return netG, netD


def human_format(num):
    """
    :param num: A number to print in a nice readable way.
    :return: A string representing this number in a readable way (e.g. 1000 --> 1K).
    """
    magnitude = 0

    while abs(num) >= 1000:
        magnitude += 1
        num /= 1000.0

    return '%.2f%s' % (num, ['', 'K', 'M', 'G', 'T', 'P'][magnitude])  # add more suffices if you need them


def print_num_params(model):
    n = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return human_format(n)


if __name__ == '__main__':
    for arch_name, s in [('FC', 64), ("DCGAN", 64), ("FastGAN", 128), ('FC', 128)]:
        netG = get_generator(arch_name, s, s)
        netD = get_discriminator(arch_name, s)
        print(arch_name)
        print("\t-G params: ", print_num_params(netG))
        print("\t-D params: ", print_num_params(netD))
