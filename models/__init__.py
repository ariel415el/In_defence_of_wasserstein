import importlib

from utils.common import parse_classnames_and_kwargs


def get_generator(args, device):
    c = 1 if args.gray_scale else 3
    model_name, kwargs = parse_classnames_and_kwargs(args.gen_arch,
                                                     kwargs={"output_dim": args.im_size, "z_dim": args.z_dim, "channels": c})
    netG = importlib.import_module("models." + model_name).Generator(**kwargs)


    print(f"G params: {print_num_params(netG)}", )

    return netG.to(device)


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
    n = sum(p.nelement() for p in model.parameters() if p.requires_grad)
    n += sum(p.nelement() for p in model.buffers() if p.requires_grad)
    return human_format(n)

