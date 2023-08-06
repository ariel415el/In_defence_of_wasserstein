import copy

import torch

from utils.common import hash_vectors


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

    def forward(self, z):
        idx = hash_vectors(z, n=self.n)
        # score = self.models[c](z)
        output = torch.cat([self.models[c](z[i].unsqueeze(0)) for i,c in enumerate(idx)], dim=0)
        return output