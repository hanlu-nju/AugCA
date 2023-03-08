import copy
from argparse import ArgumentParser, Namespace
from typing import Optional, Callable

# from skimage.filters import threshold_otsu
import matplotlib
import numpy as np
import torch
import torch.optim as optim
from pytorch_lightning import LightningModule
from torch import nn
from torch.optim import lr_scheduler, Optimizer

from model.utils import encoder_dims, get_encoder, LARS
from utils import count_acc, AverageMeter

matplotlib.use('Agg')
from tqdm import tqdm

from collections import defaultdict
from lightly.utils import dist


class ModelBase(LightningModule):
    def __init__(self, args: Namespace):
        super().__init__()
        self.args = args
        # Create encoder model
        self.backbone = self.get_encoder()
        self.accuracy_results = []
        self.running_stats = {}
        self.view_count = self.args.multi_augment + 1 if self.with_anchor() else self.args.multi_augment
        self.dataset_names = ['test', 'clf']
        self.eval_outputs = []
        self.violate_ratio = defaultdict(AverageMeter)

    def get_encoder(self):
        args = self.args
        encoder = get_encoder(args.network, args.dataset, pretrained=False)
        args.embedding_dim = args.mlp_hidden_dim = encoder_dims[args.network]
        return encoder

    def unfold_batch(self, batch):
        raise NotImplementedError

    def training_step(self, batch, batch_idx, **kwargs):
        x, class_labels = self.unfold_batch(batch)
        loss, stats = self._contrastive_loss(x, class_labels)
        self.log('loss', loss.detach().cpu().item(), on_epoch=True)
        for k, v in stats.items():
            self.log(k, v, on_epoch=True)
        return loss

    def _contrastive_loss(self, x, label_l):
        raise NotImplementedError

    def configure_optimizers(self):

        parameters = self.get_parameter_group()
        if self.args.optim == 'sgd':
            optimizer = optim.SGD(parameters, lr=self.args.lr,
                                  momentum=self.args.mo, weight_decay=self.args.wd)
        elif self.args.optim == 'adam':
            optimizer = optim.Adam(parameters, lr=self.args.lr,
                                   weight_decay=self.args.wd)
        elif self.args.optim == 'lars':
            optimizer = torch.optim.SGD(
                parameters,
                lr=self.args.lr,
                momentum=0.9,
            )
            optimizer = LARS(optimizer)
        else:
            raise RuntimeError(f'Not supported optimizer : {self.args.optim}')

        if self.args.scheduler == 'step':
            steps = [self.args.max_epochs - p for p in self.args.point]
            scheduler = lr_scheduler.MultiStepLR(optimizer, steps, gamma=self.args.gamma)
        elif self.args.scheduler == 'cosine':
            scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.args.max_epochs, eta_min=0.0)
        elif self.args.scheduler == 'constant':
            scheduler = lr_scheduler.LambdaLR(optimizer, lambda epoch: 1.0)
        else:
            raise RuntimeError(f'Not supported optimizer : {self.args.optim}')

        return [optimizer], [scheduler]

    def get_parameter_group(self):
        parameters = self.parameters()
        keys = []
        for k, v in self.named_parameters():
            keys.append(k)
        print(f'trainable parameters: {keys}')
        return parameters

    def validation_step(self, batch, batch_idx, dataloader_idx=0, *args, **kwargs):
        ret = {}
        data, label = batch
        emb = self.forward(data, emb=True)
        if dist.world_size() > 1 and dataloader_idx == 0:
            emb = torch.cat(dist.gather(emb), dim=0)
            label = torch.cat(dist.gather(label), dim=0)
        ret['label'] = label
        ret['emb'] = emb
        if hasattr(self, 'classifier'):
            logits = self.classifier(emb)
            ret['logits'] = logits
        return ret

    def validation_step_end(self, batch_parts):
        return batch_parts

    def test_step_end(self, batch_parts):
        return self.validation_step_end(batch_parts)

    def eval_knn(self, x_train, y_train, x_test, y_test, k=5):
        """ k-nearest neighbors classifier accuracy """
        d = torch.cdist(x_test, x_train)
        topk = torch.topk(d, k=k, dim=1, largest=False)
        labels = y_train[topk.indices]
        pred = torch.empty_like(y_test)
        for i in range(len(labels)):
            x = labels[i].unique(return_counts=True)
            pred[i] = x[0][x[1].argmax()]

        acc = (pred == y_test).float().mean().cpu().item() * 100
        del d, topk, labels, pred
        return acc

    def train_linear_clf(self, x_train, y_train, epoch=500):
        output_size = x_train.shape[1]
        num_class = y_train.max().item() + 1
        losses = AverageMeter()
        top1 = AverageMeter()
        with torch.enable_grad():
            lr_start, lr_end = 1e-2, 1e-6
            gamma = (lr_end / lr_start) ** (1 / epoch)
            clf = nn.Linear(output_size, num_class).to(self.device)
            optimizer = optim.Adam(clf.parameters(), lr=lr_start, weight_decay=self.args.clf_wd)
            scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
            criterion = nn.CrossEntropyLoss()
            batch_size = 1024
            num_batches = int(np.ceil(len(x_train) / batch_size))
            pbar = tqdm(range(epoch))
            for ep in pbar:
                perm = torch.randperm(len(x_train))
                for i_batch in range(num_batches):
                    idx = perm[i_batch * batch_size:(i_batch + 1) * batch_size]
                    emb_batch = x_train[idx]
                    label_batch = y_train[idx]
                    optimizer.zero_grad()
                    logits_batch = clf(emb_batch)
                    loss = criterion(logits_batch, label_batch)
                    loss.backward()
                    losses.update(loss.item(), emb_batch.size(0))
                    top1.update(count_acc(logits_batch, label_batch))
                    optimizer.step()
                pbar.set_description(f'loss: {losses.avg:.4f}, acc: {top1.avg:.4f}')
                scheduler.step()
        clf.eval()
        return clf

    def eval_classifier(self, clf, test_x, test_y, topk=(1,)):
        acc = {}
        with torch.no_grad():
            y_test_pred = clf(test_x.cuda())
        test_pred_top = y_test_pred.topk(max(topk), 1, largest=True, sorted=True).indices
        for t in topk:
            test_acc = (test_pred_top[:, :t] == test_y.cuda()[..., None]).float().sum(1).mean().cpu().item() * 100
            if t == 1:
                acc[f'accuracy'] = test_acc
            else:
                acc[f'top{t}_accuracy'] = test_acc
        return acc

    def validation_epoch_end(self, outputs):
        log_data = {
            "epoch": self.current_epoch,
        }
        if isinstance(outputs[0], dict):
            outputs = [outputs]

        dataset_names = self.dataset_names[:len(outputs)]

        embs_list, labels_list = [], []
        for i, (name, output) in enumerate(zip(dataset_names, outputs)):
            embs = torch.cat([o['emb'] for o in output])
            labels = torch.cat([o['label'] for o in output])
            embs_list.append(embs)
            labels_list.append(labels)

        clf = self.train_linear_clf(embs_list[1], labels_list[1], epoch=self.args.eval_epochs)
        for i in range(len(embs_list)):
            acc = self.eval_classifier(clf, embs_list[i], labels_list[i])
            for k, v in acc.items():
                log_data[f'{dataset_names[i]}_{k}'] = v
        if self.args.dataset != 'imagenet':
            log_data['knn_acc'] = self.eval_knn(embs_list[1], labels_list[1], embs_list[0], labels_list[0])

        for k, v in self.running_stats.items():
            log_data[k] = np.mean(v)
            v.clear()
        for k, v in log_data.items():
            self.log(k, v)
        print(log_data)

    def test_step(self, batch, batch_idx, dataloader_idx=0, *args, **kwargs):
        return self.validation_step(batch, batch_idx, dataloader_idx=dataloader_idx, *args, **kwargs)

    def test_epoch_end(self, outputs) -> None:
        self.validation_epoch_end(outputs)

    def optimizer_step(self, epoch: int = None, batch_idx: int = None, optimizer: Optimizer = None,
                       optimizer_idx: int = None, optimizer_closure: Optional[Callable] = None, on_tpu: bool = None,
                       using_native_amp: bool = None, using_lbfgs: bool = None) -> None:
        # skip the first 500 steps
        if self.args.warmup:
            if self.trainer.global_step < 500:
                lr_scale = min(1., float(self.trainer.global_step + 1) / 500.)
                for pg in optimizer.param_groups:
                    pg['lr'] = lr_scale * self.args.lr

        # update params
        optimizer.step(closure=optimizer_closure)

    @classmethod
    def add_model_specific_args(cls, parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--projection_hidden_dim', type=int, default=1024)
        parser.add_argument('--projection_output_dim', type=int, default=64)
        parser.add_argument('--projection_mlp_layers', type=int, default=2)
        parser.add_argument('--mlp_normalization', type=str, default='bn')
        parser.add_argument('--temperature', type=float, default=0.5)
        parser.add_argument('--metric', type=str, default='cosine')
        parser.add_argument('--clf_wd', type=float, default=5e-6)
        return parser

    @classmethod
    def model_type(cls):
        raise NotImplementedError

    @property
    def model_name(self):
        return f'{self.__class__.__name__}-t{self.args.temperature}'

    @classmethod
    def multi_augment(cls, args) -> int:
        raise NotImplementedError

    @classmethod
    def with_anchor(cls) -> bool:
        return False
