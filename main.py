from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger

from data.datasets import get_data_module
from utils import *
from model import model_dict


def main():
    parser = ArgumentParser()

    # add model specific args
    parser = add_arguments(parser)

    temp_args, _ = parser.parse_known_args()
    model = model_dict[temp_args.model]
    parser = model.add_model_specific_args(parser)
    # add all the available trainer options to argparse
    # ie: now --gpus --num_nodes ... --fast_dev_run all work in the cli
    args = parser.parse_args()

    set_random_seed(args.seed)

    debug = is_debug()
    model_class = model_dict[args.model]
    set_gpu(args)

    args.type = model_class.model_type()
    if not hasattr(args, 'multi_augment'):
        args.multi_augment = model_class.multi_augment(args)
    args.with_anchor = model_class.with_anchor()
    data_module = get_data_module(args)
    model = model_class(args)
    if debug:
        print('In debug mode, disable logger')
        logger = False
        debug_params = {
            'limit_val_batches': 5,
            'max_epochs': 1,
        }
    else:
        print('In release mode, enable logger')
        name = f'{args.model}-{args.network}-{args.dataset}'
        project_name = f'ICLR2023'
        try:
            # try to log with wandb
            import wandb
            logger = WandbLogger(
                project=project_name,
                name=name,
            )
        except ImportError:
            # use tensorboard
            logger = TensorBoardLogger(save_dir="tb_logs", name=f"{project_name}-{name}")
        debug_params = {'max_epochs': args.max_epochs,
                        'weights_save_path': f'checkpoint/{model.model_name}/'}

    trainer = Trainer(gpus=len(args.gpus.split(',')), strategy='ddp_find_unused_parameters_false', logger=logger,
                      check_val_every_n_epoch=args.val_every_n_epoch, num_sanity_val_steps=2,
                      log_every_n_steps=4, sync_batchnorm=True,
                      **debug_params)
    print(vars(args))
    if args.init_weights is not None:
        model_load_weights(args.init_weights, model)
    print(model.model_type())
    if args.eval:
        trainer.test(model, data_module)
    else:
        trainer.fit(model, data_module)
        trainer.test(datamodule=data_module)


if __name__ == '__main__':
    main()
