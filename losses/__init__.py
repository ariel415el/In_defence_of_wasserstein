import sys

from losses.c_transform_wgan import *
from losses.non_saturating import *
from losses.optimal_transport import *
from losses.soft_hinge_loss import *
from losses.two_steps_loss import *
from losses.wgan import *

def get_loss_function(loss_name):
    loss_name_and_args = loss_name.split('-')
    loss_name = loss_name_and_args[0]
    kwargs = dict()
    for arg in loss_name_and_args[1:]:
        name, value = arg.split("=")
        kwargs[name] = value
    return getattr(sys.modules[__name__], loss_name)(**kwargs)