
from argparse import Namespace

import torch
import torch.nn.functional as F
from lightly.utils import dist
from torch import nn

from .simclr import SimCLR


class ACAPCLoss(nn.Module):

    def __init__(self, gather_distributed: bool = False, K=1, temperature=0.5):
        super().__init__()
        self.K = K
        self.gather_distributed = gather_distributed
        self.temperature = temperature

    def forward(self,
                out0: torch.Tensor,
                out1: torch.Tensor):
        """Forward pass through ACA-PC Loss.

        Args:
            out0:
                Output projections of the first set of transformed images.
                Shape: (batch_size, embedding_size)
            out1:
                Output projections of the second set of transformed images.
                Shape: (batch_size, embedding_size)

        Returns:
            ACA-PC Loss value.

        """

        device = out0.device
        batch_size, _ = out0.shape

        # normalize the output to length 1
        out0 = nn.functional.normalize(out0, dim=1)
        out1 = nn.functional.normalize(out1, dim=1)

        if self.gather_distributed and dist.world_size() > 1:
            # gather hidden representations from other processes
            out0_large = torch.cat(dist.gather(out0), 0)
            out1_large = torch.cat(dist.gather(out1), 0)
            diag_mask = dist.eye_rank(batch_size, device=out0.device)
        else:
            # single process
            out0_large = out0
            out1_large = out1
            diag_mask = torch.eye(batch_size, device=out0.device, dtype=torch.bool)

        # calculate similiarities
        # here n = batch_size and m = batch_size * world_size
        # the resulting vectors have shape (n, m)
        logits_00 = torch.einsum('nc,mc->nm', out0, out0_large) / self.temperature
        logits_01 = torch.einsum('nc,mc->nm', out0, out1_large) / self.temperature
        logits_10 = torch.einsum('nc,mc->nm', out1, out0_large) / self.temperature
        logits_11 = torch.einsum('nc,mc->nm', out1, out1_large) / self.temperature

        # remove simliarities between same views of the same image
        logits_00 = logits_00[~diag_mask].view(batch_size, -1)
        logits_11 = logits_11[~diag_mask].view(batch_size, -1)

        # concatenate logits
        # the logits tensor in the end has shape (2*n, 2*m-1)
        logits_0100 = torch.cat([logits_01, logits_00], dim=1)
        logits_1011 = torch.cat([logits_10, logits_11], dim=1)
        logits = torch.cat([logits_0100, logits_1011], dim=0)

        # create labels
        labels = torch.arange(batch_size, device=device, dtype=torch.long)
        labels = labels + dist.rank() * batch_size
        labels = labels.repeat(2)

        ############### SimCLR Loss #################
        # loss = self.cross_entropy(logits, labels) #
        #############################################

        pos_mask = F.one_hot(labels, num_classes=logits.size(-1)).bool()

        pos = torch.mean(logits[pos_mask])
        neg = torch.mean(logits[~pos_mask] ** 2)

        loss = -2 * pos + self.K * neg

        return loss


class ProjectionLoss(nn.Module):

    def __init__(self, normalize=False):
        super().__init__()
        self.normalize = normalize

    def forward(self, out0: torch.Tensor,
                out1: torch.Tensor,
                out: torch.Tensor):
        target = 0.5 * (out0 + out1)
        if self.normalize:
            target = F.normalize(target, dim=-1)
            out = F.normalize(out, dim=-1)
        return F.mse_loss(out, target.detach())


class AugmentationComponentAnalysis(SimCLR):

    @classmethod
    def multi_augment(cls, args) -> int:
        return 2

    def __init__(self, args: Namespace):
        super().__init__(args)
        self.pc_loss = ACAPCLoss(gather_distributed=True, K=self.args.K)
        self.proj_loss = ProjectionLoss(normalize=self.args.proj_norm)

    def _contrastive_loss(self, x, label_l):
        """Compute loss for model.

        Args:
            k: hidden vector of shape [bsz, ...].
            q: hidden vector of shape [bsz, ...].
            y: ground truth of shape [bsz].
        Returns:
            A loss scalar.
        """
        x0, x1, x_ = x[:, 1], x[:, 2], x[:, 0]
        z0 = self.forward(x0)
        z1 = self.forward(x1)
        z_ = self.forward(x_)
        pc_loss = self.pc_loss(z0, z1)
        proj_loss = self.proj_loss(z0, z1, z_)
        self.log('pc_loss', pc_loss)
        self.log('proj_loss', proj_loss)
        loss = pc_loss + self.args.lamb * proj_loss
        stats = {}
        return loss, stats

    @classmethod
    def add_model_specific_args(cls, parent_parser):
        parser = super().add_model_specific_args(parent_parser)
        parser.add_argument('--K', type=float, default=1.0)
        parser.add_argument('--lamb', type=float, default=1.0)
        parser.add_argument('--proj_norm', action='store_true')
        return parser

    @classmethod
    def with_anchor(cls) -> bool:
        return True
