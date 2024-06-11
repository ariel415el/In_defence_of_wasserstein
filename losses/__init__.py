import os
import sys

from losses.c_transform_wgan import *
from losses.non_saturating import *
from losses.optimal_transport import *
from losses.soft_hinge_loss import *
# from losses.two_steps_loss import *
from losses.wgan import *
from losses.adverserial_feature_matching import *

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "utils"))
from common import parse_classnames_and_kwargs
# from utils.common import parse_classnames_and_kwargs


def get_loss_function(loss_description):
    loss_name, kwargs = parse_classnames_and_kwargs(loss_description)
    loss = getattr(sys.modules[__name__], loss_name)(**kwargs)
    loss.name = loss_description
    return loss
