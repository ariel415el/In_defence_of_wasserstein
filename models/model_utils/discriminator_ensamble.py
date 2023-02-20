import torch
import copy
import numpy as np


class Ensemble(torch.nn.Module):
    def __init__(self, model, n):
        self.n = n
        super(Ensemble, self).__init__()
        self.models = []
        for _ in range(n):
            new_model = copy.deepcopy(model)
            for layer in new_model.children():
                if hasattr(layer, 'reset_parameters'):
                    layer.reset_parameters()

            self.models.append(new_model)

        self.models = torch.nn.ModuleList(self.models)

    def forward(self, x, return_std=False):
        scores = torch.stack([model(x) for model in self.models])
        score = scores.mean(0)
        if return_std:
            std = scores.std(0).mean().item()
            norm = scores.abs().mean().item()
            return score, std / norm
        else:
            return score


class StochasticEnsemble(torch.nn.Module):
    def __init__(self, model, n):
        self.n = n
        super(StochasticEnsemble, self).__init__()
        self.models = []
        for _ in range(n):
            new_model = copy.deepcopy(model)
            for layer in new_model.children():
                if hasattr(layer, 'reset_parameters'):
                    layer.reset_parameters()

            self.models.append(new_model)

        self.models = torch.nn.ModuleList(self.models)

    def forward(self, x, return_std=False):
        i = np.random.randint(self.n)
        score = self.models[i](x)
        if return_std:
            with torch.no_grad():
                scores = torch.stack([model(x) for model in self.models])
                std = scores.std(0).mean().item()
                norm = scores.abs().mean().item()
            return score, std / norm
        else:
            return score
