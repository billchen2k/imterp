
import argparse
import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import yaml
from einops import asnumpy, rearrange, repeat
from jsonargparse import ActionConfigFile, ArgumentParser
from torch.utils.tensorboard import SummaryWriter

from data_loader import GKDataLoader, load_funcs
from model.gknet import GKNet
from utils import (ArgparseFormatter, astensor, datestr, logger, mse, outdir,
                   read_config)


class Trainer(object):

    def __init__(self, args) -> None:
        self.args = args
        self.device = th.device(read_config('device'))
        self.save_epoch = read_config('save_epoch')
        self.loader = GKDataLoader(dataset=args.dataset,
                                   batch_size=args.batch_size,
                                   p=args.p,
                                   max_nodes=args.max_nodes,
                                   temporal_sr=args.t_sr,
                                   unknown_rate=args.unknown_rate,
                                   masked_rate=args.masked_rate,
                                   train_rate=args.train_rate,
                                   adj_k=args.k)
        self.model = GKNet(in_size=1,
                           info=self.loader.info,
                           temporal_size=args.p,
                           temporal_sr=args.t_sr,
                           hidden_size=args.z,
                           t_kernel_size=args.wt,
                           pe_size=args.pe,
                           t_dilation=1,
                           device=self.device,
                           spec=args.spec,
                           dropout=args.dropout)
        self.hash = f'imterp_{datestr()}_{args.dataset}_k{args.k}_p{args.p}_z{args.z}_wt{args.wt}_pe{args.pe}_tsr_{args.t_sr}'
        self.writer = SummaryWriter(comment=self.hash, log_dir=outdir(f'./results/{self.hash}'))
        self.writer.add_text('args', str(args))
        with open(outdir(f'./results/{self.hash}/train_args.yaml'), 'w') as f:
            yaml.dump(vars(args), f, default_flow_style=False)
        joblib.dump(self.loader.coords, outdir(f'./results/{self.hash}/train/coords.z'))
        logger.info(f'Trainer loaded: {self.hash}')

    def train(self):
        logger.info(f'Training started.')

        ag = self.args
        optimizer = th.optim.Adam(self.model.parameters(), lr=ag.lr)

        self.model.train()

        for e in range(ag.epoch + 1):
            epoch_losses = []
            for i in range(self.loader.train_total_t // (ag.p * ag.batch_size)):
                # for i in range(self.loader.train_total_t // (ag.batch_size)):
                X_batch, Y_batch, A_first, A_sub, masked_set, coords = self.loader.sample()

                X_batch = astensor(X_batch)
                input_flag = th.ones_like(astensor(Y_batch))
                if ag.ignore0:
                    input_flag = th.where(X_batch == 0, 0, 1)

                optimizer.zero_grad()
                X_batch_predict = self.model.forward(
                    astensor(X_batch),
                    astensor(A_first),
                    astensor(A_sub),
                    astensor(coords),
                )

                # only calculate the loss of masked nodes
                loss_flag = th.zeros_like(astensor(Y_batch)).to(self.device).float()
                loss_flag[:, :, list(masked_set), :] = 1


                X_predict = X_batch_predict * loss_flag * input_flag
                Y_batch = astensor(Y_batch)
                loss_mse = nn.MSELoss()(X_predict, Y_batch)

                # as_log_softmax = lambda a:  F.log_softmax(rearrange(a, 'b c n p -> b (c n p)'), dim=1)
                # as_softmax = lambda a: F.softmax(rearrange(a, 'b c n p -> b (c n p)'), dim=1)

                loss_huber = nn.HuberLoss(delta=0.1)(X_predict, Y_batch)
                loss = loss_huber

                loss.backward()
                optimizer.step()

                epoch_losses.append(loss.item())

            rmse, mae, mape, rmse_uct = self.eval(e)
            loss_val = np.mean(epoch_losses)
            self.writer.add_scalar('train/loss', loss_val, e)
            self.writer.add_scalar('eval/rmse', rmse, e)
            self.writer.add_scalar('eval/mae', mae, e)
            self.writer.add_scalar('eval/mape', mape, e)
            self.writer.add_scalar('eval/rmse_uncertainty', rmse_uct, e)
            logger.debug(f'epoch {e} - loss: {loss_val:.7f}, rmse: {rmse:.6f}, mae: {mae:.6f}, mape: {mape:.6f}')

            if e % (self.save_epoch // 2) == 0:
                self.writer.add_image(
                    'train/output',
                    rearrange(X_batch_predict, 'b c n p -> c n (b p)'),
                    e,
                )
            self.writer.add_image('train/truth', rearrange(Y_batch, 'b c n p -> c n (b p)'), e)

            if e % self.save_epoch == 0:
                th.save(
                    self.model.state_dict(),
                    outdir(f'./results/{self.hash}/checkpoints/e{e}_loss{loss_val:.5f}_rmse{rmse:.5f}.pth'))

    def eval(self, e: int):
        self.model.eval()

        with th.no_grad():
            # X: [groups, batch, 1, num_nodes, p]
            X_batch_groups, Y_batch_groups, A_first, A_sub, coords = self.loader.sample_eval()
            X_batch_groups, Y_batch_groups = astensor(X_batch_groups, Y_batch_groups)
            X_predict = th.zeros_like(Y_batch_groups)
            for g in range(X_batch_groups.shape[0]):
                X_predict[g, :, :, :, :] = self.model.forward(
                    X_batch_groups[g, :, :, :, :],
                    astensor(A_first),
                    astensor(A_sub),
                    astensor(coords),
                )

            # Only calculate the loss for unknown nodes
            loss_flag = th.zeros_like(Y_batch_groups)
            loss_flag[:, :, :, list(self.loader.unknown_set), :] = 1
            # loss_flag = repeat(loss_flag, 'b c n p -> g b c n p', g=X_batch_groups.shape[0])

            # Only calculate uncertainty for known nodes.
            # @Deprecated! not using.
            uncertainty_flag = th.ones_like(Y_batch_groups)
            uncertainty_flag[:, :, :, list(self.loader.unknown_set), :] = 0

            input_flag = th.ones_like(Y_batch_groups)
            # if self.args.ignore0:
            #     input_flag = th.where(X_batch_groups == 0, 0, 1)

            X_predict = self.loader.scaler.inv(X_predict)
            Y_batch_groups = self.loader.scaler.inv(Y_batch_groups)

            X_predict_image = rearrange(X_predict, 'g b c n p -> c n (g b p)')
            Y_batch_image = rearrange(Y_batch_groups, 'g b c n p -> c n (g b p)')
            e == 0 and self.writer.add_image('eval/truth', self.loader.scaler.norm(Y_batch_image), e)

            if e % (self.save_epoch // 2) == 0:
                self.writer.add_image('eval/output', self.loader.scaler.norm(X_predict_image), e)

            if e % self.save_epoch == 0:
                # Persist output values for testing
                joblib.dump(asnumpy(rearrange(X_predict, 'g b 1 n p -> n (g b p)')),
                            outdir(f'./results/{self.hash}/train/output_e{e}.z'))
                joblib.dump(asnumpy(rearrange(Y_batch_groups, 'g b 1 n p -> n (g b p)')),
                            outdir(f'./results/{self.hash}/train/truth.z'))
                joblib.dump(set(self.loader.unknown_set), outdir(f'./results/{self.hash}/train/unknown_set.z'))

            truth = Y_batch_groups * loss_flag * input_flag
            pred = X_predict * loss_flag * input_flag
            rmse = mse(pred, truth, squared=False)
            mae = th.abs(pred - truth).mean()
            # mape = th.abs(pred - truth).mean() / truth.mean()
            mape = th.abs((pred - truth) / truth).mean()
            mape = th.abs((th.masked_select(pred, truth != 0) - th.masked_select(truth, truth != 0)) /
                        th.masked_select(truth, truth != 0)).mean()

            measurement = Y_batch_groups * uncertainty_flag * input_flag
            pred_with_uncert = X_predict * uncertainty_flag * input_flag
            rmse_uct = mse(pred_with_uncert, measurement, squared=False)

            return rmse, mae, mape, rmse_uct


if __name__ == '__main__':

    parser = argparse.ArgumentParser(formatter_class=ArgparseFormatter,
                                     description='''
       Temporal ────▶
Spatial┌───────────────────────────────┬──────────┐
   │   │                        masked │          │
   │   ├ ─ ─ ─ Train ─ ─ ─ ─ ─ ─ ─ ─ ─ ┤          │
   ▼   │     Masked ->                 │   known  │  known_rate
       │  Masked + Visible     visible │          │
       │                               │          │
       ├───────────────────────────────┼──────────┤
       │             Test              │   eval   │
       │           test loss           │   loss   │ unknown_rate
       └───────────────────────────────┴──────────┘
                  train_rate               Eval
''')
    parser.add_argument('--epoch', type=int, default=800)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--dataset', type=str, default='ushcn', help=f'Dataset, one of {list(load_funcs.keys())}')

    parser.add_argument('--k', type=int, default=9, help='Top k for spatial convolution layers.')
    parser.add_argument('--z', type=int, default=24, help='Hidden space dimension.')
    parser.add_argument('--p', type=int, default=9, help='Temporal window length.')
    parser.add_argument('--pe', type=int, default=0, help='Positional encoding dimesion. 0 to turn off.')
    parser.add_argument('--pe_scales',
                        type=int,
                        default=8,
                        help='Number of scales (frequencies) used in the positional encodin.')
    parser.add_argument('--wt', type=int, default=3, help='Temporal conv kernel size.')
    parser.add_argument('--t_sr', type=int, default=2, help='Temporal super resolution rate.')

    parser.add_argument('--spec',
                        type=str,
                        default='STSTr',
                        help='Model specification. S = Spatial Conv, T = Temporal Conv, r = relu act, t = tanh act')
    parser.add_argument('--ignore0', type=bool, default=True, help='Ignore zero values in the original datasets.')
    parser.add_argument('--max_nodes',
                        type=int,
                        default=1500,
                        help='Max number of nodes used in the dataset. If larger, will sample.')
    parser.add_argument('--unknown_rate',
                        type=float,
                        default=0.4,
                        help='Ratio of unknown nodes. Will be invisible during training.')
    parser.add_argument('--masked_rate',
                        type=float,
                        default=0.4,
                        help='Ratio of masked nodes during training. Will be invisible during the training batch.')
    parser.add_argument('--train_rate',
                        type=float,
                        default=0.75,
                        help='Ratio of timesteps used for training. Rest timesteps will be used for testing.')

    parser.add_argument('--note', type=str, default='', help='Additional notes for this run.')

    # Use this key to specify all other parameters.
    parser.add_argument('--config', '-c', type=str, default='train.ushcn', help='Specify configration to use inside config.yaml.')

    args = parser.parse_args()

    if args.config:
        config_args = read_config(args.config)
        vars(args).update(config_args)

    trainer = Trainer(args)

    import shutil
    import signal
    import sys

    def signal_handler(sig, frame):
        response = input("Do you want to delete the runs? (y/n): ")
        if response.lower() == "y":
            shutil.rmtree(outdir(f'./results/{trainer.hash}'))
            # shutil.rmtree(outdir(f'./checkpoints/{trainer.hash}'))
            # shutil.rmtree(outdir(f'{trainer.writer.log_dir}'))
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    trainer.train()
