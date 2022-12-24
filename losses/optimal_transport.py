import numpy as np
import ot
import torch

from losses.loss_utils import vgg_dist_calculator


class BatchW2D:
    """
    """
    def __init__(self, search_space='full'):
        self.dist_clc = vgg_dist_calculator()
        self.search_space = search_space

    def trainG(self, netD, real_data, fake_data):
        uniform_x = np.ones(len(real_data)) / len(real_data)
        uniform_y = np.ones(len(fake_data)) / len(fake_data)
        with torch.no_grad():
            # C = torch.mean((real_data[:, None] - fake_data[None, :]) ** 2, dim=(-3, -2,-1))
            C = self.dist_clc.get_dist_mat(real_data, fake_data, b=64)
            C = C.cpu().numpy()
            Tmap = ot.emd(uniform_x, uniform_y, C)
            Nns = Tmap.argmax(0)
            OT = np.sum(Tmap * C)

        Gloss = ((fake_data - real_data[Nns])**2).mean()

        return Gloss, {"Gloss": Gloss.item(), "OT": OT.item()}
