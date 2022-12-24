import sys

from losses.c_transform_wgan import *
from losses.non_saturating import *
from losses.optimal_transport import *
from losses.soft_hinge_loss import *
from losses.two_steps_loss import *
from losses.wgan import *

def get_loss_function(loss_name):
    return getattr(sys.modules[__name__], loss_name)()