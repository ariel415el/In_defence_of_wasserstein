from losses.loss_utils import calc_gradient_penalty


class WGANLoss:
    """Should be used with GP"""
    def trainD(self, netD, real_data, fake_data):
        real_score = netD(real_data).mean()
        fake_score = netD(fake_data.detach()).mean()
        WD = real_score - fake_score
        Dloss = -1 * WD  # Maximize WD

        debug_dict = {"W1 ": WD.item()}
        return Dloss, debug_dict

    def trainG(self, netD, real_data, fake_data):
        Gloss = -1* netD(fake_data).mean() #  Minimize WD
        return Gloss, {"Gloss": Gloss.item()}
