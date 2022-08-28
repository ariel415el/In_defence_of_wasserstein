import torch
import torch.nn.functional as F


def patch_swd(x, y, patch_size=7, num_proj=1024):
    with torch.no_grad():
        b, c, h, w = x.shape

        # Sample random normalized projections
        rand = torch.randn(num_proj, c * patch_size ** 2).to(x.device)  # (slice_size**2*ch)
        rand = rand / torch.norm(rand, dim=1, keepdim=True)  # noramlize to unit directions
        rand = rand.reshape(num_proj, c, patch_size, patch_size)

        # Project patches
        projx = F.conv2d(x, rand).transpose(1, 0).reshape(b, num_proj, -1)
        projy = F.conv2d(y, rand).transpose(1, 0).reshape(b, num_proj, -1)

        # Sort and compute L1 loss
        projx, _ = torch.sort(projx, dim=-1)
        projy, _ = torch.sort(projy, dim=-1)

        loss = torch.abs(projx - projy).mean()

        return loss


def swd(x, y, num_proj=1024):
    with torch.no_grad():
        b, c, h, w = x.shape

        # Sample random normalized projections
        rand = torch.randn(num_proj, c * h * w).to(x.device)  # (slice_size**2*ch)
        rand = rand / torch.norm(rand, dim=1, keepdim=True)  # noramlize to unit directions
        # rand = rand.reshape(self.num_proj, c, h, )

        # Sort and compute L1 loss
        projx = x.reshape(b, -1) @ rand.T
        projy = y.reshape(b, -1) @ rand.T

        loss = torch.abs(projx - projy).mean()

        return loss
