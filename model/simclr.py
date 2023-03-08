
# import wandb
from argparse import Namespace

from .base import ModelBase
from lightly.loss import NTXentLoss
from lightly.models.modules import SimCLRProjectionHead


class SimCLR(ModelBase):

    @classmethod
    def multi_augment(cls, args) -> int:
        return 2

    @classmethod
    def add_model_specific_args(cls, parent_parser):
        parser = super().add_model_specific_args(parent_parser)
        parser.add_argument('--multi_augment', type=int, default=2)
        return parser

    @property
    def model_name(self):
        return f'{self.__class__.__name__}-{self.args.projection_mlp_layers}-{self.args.projection_hidden_dim}-' \
               f'{self.args.projection_output_dim}'

    @classmethod
    def model_type(cls):
        return 'unsup'

    def __init__(self, args: Namespace):
        super().__init__(args)
        self.projection_head = SimCLRProjectionHead(input_dim=args.embedding_dim,
                                                    hidden_dim=args.projection_hidden_dim,
                                                    output_dim=args.projection_output_dim)
        self.criterion = NTXentLoss(gather_distributed=True)

    def unfold_batch(self, batch):
        return batch

        # self.log('loss', loss.item(), on_epoch=True)
        # return loss

    def forward(self, x, emb=False):
        x = self.backbone(x).flatten(start_dim=1)
        if emb:
            return x
        z = self.projection_head(x)
        return z

    def _contrastive_loss(self, x, label):
        """Compute loss for model.

        Args:
            k: hidden vector of shape [bsz, ...].
            q: hidden vector of shape [bsz, ...].
            y: ground truth of shape [bsz].
        Returns:
            A loss scalar.
        """
        # ims = torch.cat([x[:, i].contiguous() for i in range(self.view_count)])
        x0, x1 = x[:, 0], x[:, 1]
        z0 = self.forward(x0)
        z1 = self.forward(x1)
        loss = self.criterion(z0, z1)
        stats = {}
        return loss, stats
