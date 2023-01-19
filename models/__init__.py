import importlib

from utils.common import parse_classnames_and_kwargs


def get_models(args, device):
    model_name, kwargs = parse_classnames_and_kwargs(args.gen_arch, kwargs={"output_dim": args.im_size, "z_dim": args.z_dim})
    netG = importlib.import_module("models." + model_name).Generator(**kwargs)

    model_name, kwargs = parse_classnames_and_kwargs(args.disc_arch, kwargs={"input_dim": args.im_size})
    netD = importlib.import_module("models." + model_name).Discriminator(**kwargs)

    if args.spectral_normalization:
        from models.model_utils import make_model_spectral_normalized
        netD = make_model_spectral_normalized(netD)

    print(f"G params: {print_num_params(netG)}, D params: {print_num_params(netD)}", )

    return netG.to(device), netD.to(device)


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
    for arch_name, s in [('FC', 64), ("DCGAN", 64), ('ResNet', 64),
                         ('FC', 128), ("DCGAN", 128),('ResNet', 128) , ("FastGAN", 128), ('StyleGAN', 128),
                         ('BagNet-9', 64), ('BagNet-9', 128)]:
        print(f"{arch_name}: {s}x{s}")
        try:
            netG = get_generator(arch_name, s, s)
            print("\t-G params: ", print_num_params(netG))
        except Exception as e:
            pass
        netD = get_discriminator(arch_name, s)
        print("\t-D params: ", print_num_params(netD))
