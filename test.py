import gc
import json
import os
from copy import deepcopy
from typing import Dict, Iterable, Union

import bottleneck as bn
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch as th
import yaml
from einops import asnumpy, rearrange, repeat
from joblib import delayed
from joblib.parallel import Parallel
from jsonargparse import ArgumentParser, Namespace

from data_loader import GKDataLoader, compute_A
from imterp.interpolator import Interpolator
from model.gknet import GKNet
from utils import (NpEncoder, adj_top_k, astensor, datestr, logger, mse, npsave, outdir, read_config, ssim)

NUM_PROCESS = 12

class ImTerp(object):

    def __init__(self, args) -> None:
        self.device = read_config('device')
        train_args = yaml.load(open(f'{args.workdir}/train_args.yaml'), Loader=yaml.FullLoader)
        train_args = Namespace(**train_args)
        logger.info(f'Training args: {train_args}')
        logger.info(f'Working directory: {args.workdir}')
        self.train_args = train_args
        self.loader = GKDataLoader(
            dataset=train_args.dataset,
            batch_size=train_args.batch_size,
            p=train_args.p,
            max_nodes=train_args.max_nodes,
            temporal_sr=train_args.t_sr,
            unknown_rate=train_args.unknown_rate,
            masked_rate=train_args.masked_rate,
            train_rate=train_args.train_rate,
            adj_k=train_args.k,
            outdir=args.workdir,
        )
        self.model = GKNet(
            in_size=1,
            out_size=1,
            info=self.loader.info,
            temporal_size=train_args.p,
            temporal_sr=train_args.t_sr,
            hidden_size=train_args.z,
            t_kernel_size=train_args.wt,
            pe_size=train_args.pe,
            t_dilation=1,
            device=self.device,
            spec=train_args.spec,
            dropout=train_args.dropout,
            nopna=train_args.nopna if 'nopna' in train_args else False,
        )

        all_checkpoints = os.listdir(f'{args.workdir}/checkpoints')
        checkpoints = list(filter(lambda x: x.startswith(args.checkpoint), all_checkpoints))
        if len(checkpoints) == 0:
            logger.error(f'Checkpoint {args.checkpoint} not found.')
            exit(-1)
        checkpoint = list(checkpoints)[0]
        logger.info(f'Will use checkpoint f{checkpoint}...')
        self.model.load_state_dict(th.load(f'{args.workdir}/checkpoints/{checkpoint}'))

        interp_config = read_config(f'interp.{train_args.dataset}')
        self.interp_method = interp_config['method']
        self.interp_params = interp_config['params']

        self.args = args

    def _get_model_output(self, X_batch_groups: np.ndarray, A_first: np.ndarray, A_sub: np.ndarray,
                          coords: np.ndarray) -> np.ndarray:
        """Get predict value from group batch data.

        Args:
            X_batch_groups (np.ndarray): Group data of shape
            A_first (np.ndarray): Adjacency matrix of the first spatial conv
            A_sub (np.ndarray): Adjacancy matrix of subsequent spatial conv
            coords (np.ndarray): Coordinates of shape [num_nodes, 2]

        Returns:
            np.ndarray: _description_
        """
        self.model.eval()
        predict = astensor(np.zeros_like(repeat(X_batch_groups, 'g b 1 n p -> g b 1 n (p tsr)', tsr=self.train_args.t_sr)))
        with th.no_grad():
            for g in range(X_batch_groups.shape[0]):
                predict[g, :, :, :, :] = self.model.forward(
                    astensor(X_batch_groups[g, :, :, :, :]),
                    astensor(A_first),
                    astensor(A_sub),
                    astensor(coords),
                )
        output = rearrange(predict, 'g b 1 n p -> n (g b p)')
        output = asnumpy(output)
        logger.debug(f'Model output: {output.shape}')
        return output

    # def plot_interp(self, data: np.ndarray, outdir: string, plot_bg: bool = False, plot_scatter: bool = False):

    def _interp_worker(self, t, tsr_id, plot_bg: bool, plot_scatter: bool) -> Dict:
        logger.debug(f'Interpolating worker t={t}, tsr_id={tsr_id}...')
        scaler = self.loader.scaler
        args = self.args

        tsr = self.train_args.t_sr
        tt = t * tsr + tsr_id

        if tsr > 1:
            # if tsr > 1, the masked map is linear interpolation result
            v_left = self.loader.X_eval_all[list(self.loader.known_set), t]
            v_right = self.loader.X_eval_all[list(self.loader.known_set), (t + 1)]
            weight_right = (tsr_id + 1) / (tsr + 1)
            v_linear_interp = v_left * (1 - weight_right) + v_right * weight_right
            masked = Interpolator(sensor_coords=self.loader.coords[list(self.loader.known_set), :],
                                  values=scaler.inv(v_linear_interp),
                                  terrain_gdf=self.loader.terrain_gdf,
                                  grid_size=self.args.grid_size,
                                  params=self.interp_params)
        else:
            # regular spatial test
            masked = Interpolator(sensor_coords=self.loader.coords[list(self.loader.known_set), :],
                                values=scaler.inv(self.loader.X_eval_all[list(self.loader.known_set), tt]),
                                terrain_gdf=self.loader.terrain_gdf,
                                grid_size=self.args.grid_size,
                                params=self.interp_params)
        z_masked = masked.interp(self.interp_method)

        densified = Interpolator(sensor_coords=self.densified_coords,
                                 values=self.X_predict[:, tt],
                                 terrain_gdf=self.loader.terrain_gdf,
                                 grid_size=self.args.grid_size,
                                 params=self.interp_params)
        z_densified = densified.interp(self.interp_method)

        truth = Interpolator(sensor_coords=self.loader.coords,
                             values=scaler.inv(self.loader.X_eval_all[:, tt]),
                             terrain_gdf=self.loader.terrain_gdf,
                             grid_size=self.args.grid_size,
                             params=self.interp_params)
        z_truth = truth.interp(self.interp_method)

        mse_masked = mse(z_truth, z_masked)
        mse_densified = mse(z_truth, z_densified)
        ssim_masked = ssim(z_truth, z_masked)
        ssim_densified = ssim(z_truth, z_densified)
        zs = [z_masked, z_densified, z_truth]
        vrange = [min([bn.nanmin(z) for z in zs]), max([bn.nanmax(z) for z in zs])]
        vrange = np.round(vrange, 2)

        if tsr > 1:
            vrange = np.array([0, 130])
        logfunc = logger.success if mse_densified < mse_masked else logger.warning
        logfunc(
            f't={t}, tsrid={tsr_id}, MSE: masked={mse_masked:.5f}, densified={mse_densified:.5f}, SSIM: masked={ssim_masked:.4f}, densified={ssim_densified:.4f}, vrange={vrange}'
        )

        outprefix = f'{datestr()[-4:]}_t{t}.{tsr_id}_dr{args.densify_ratio}'
        if args.plot:
            masked.plot_result(
                known_coords=list(range(len(self.loader.known_set))),
                vrange=vrange,
                outdir=outdir(
                    f'{args.workdir}/interp/{outprefix}_masked_{mse_masked:.5f}{"V" if mse_densified < mse_masked else ""}.png'
                ),
                title=f'{outprefix}_masked_mse_{mse_masked:.5f}',
                plot_bg=plot_bg,
                plot_scatter=plot_scatter)
            densified.plot_result(
                known_coords=[i for i in range(self.densified_coords.shape[0]) if i not in self.loader.densified_set],
                vrange=vrange,
                outdir=outdir(f'{args.workdir}/interp/{outprefix}_densified_{mse_densified:.5f}.png'),
                title=f'{outprefix}_densified_mse_{mse_densified:.5f}',
                plot_bg=plot_bg,
                plot_scatter=plot_scatter)
            truth.plot_result(known_coords=list(range(len(self.loader.coords))),
                              vrange=vrange,
                              outdir=outdir(f'{args.workdir}/interp/{outprefix}_truth.png'),
                              title=f'{outprefix}_truth',
                              plot_bg=plot_bg,
                              plot_scatter=plot_scatter)

        del masked, densified, truth

        # Save for visualization
        readable_date = self.dates[t].strftime('%Y%m%d%H%M%S')
        if self.train_args.t_sr > 1:
            readable_date += f' / {tsr_id}'

        names = [
            f't{t}_{readable_date}_masked',
            f't{t}_{readable_date}_densified',
            f't{t}_{readable_date}_truth',
        ]

        data_to_save = [np.flip(z_masked, axis=0), np.flip(z_densified, axis=0), np.flip(z_truth, axis=0)]
        for data, name in zip(data_to_save, names):
            npsave(data, f'{self.args.workdir}/vis/maps/{name}.npy', with_png=True)

        gc.collect()
        return {
            't': tt,
            'tsr_id': tsr_id,
            'date': self.dates[t].isoformat(),
            'mse_masked': mse_masked,
            'mse_densified': mse_densified,
            'ssim_masked': 0 if np.isnan(ssim_masked) else ssim_masked,
            'ssim_densified': 0 if np.isnan(ssim_densified) else ssim_densified,
            'diff': mse_masked - mse_densified,  # > 0 is good
            'map_masked': f'/maps/{names[0]}.npy',
            'map_densified': f'/maps/{names[1]}.npy',
            'map_truth': f'/maps/{names[2]}.npy',
            'vrange': vrange.tolist(),
            'densified_vrange': [bn.nanmin(z_densified), bn.nanmax(z_densified)],
        }

    def interp(
        self,
        t_range: Union[Iterable[int], int] = [0, 10, 1],  # [start, end, step] or int
        # source: Union[Iterable[int], str] = 'knwon',  # 'all', 'known', all a list of index
        # tsr_linear: bool=False # if enabled, the masked data will be used for linear t_sr. For experiment only.
    ) -> None:

        # self.tsr_linear = tsr_linear
        if self.train_args.t_sr > 1:
            logger.warning(f'Temporal SR emabled (x{self.train_args.t_sr}). Masked data will be linear interpolation.')

        self.model.eval()
        X_batch_groups, Y_batch_groups, A_first, A_sub, densified_coords, dates = self.loader.sample_densify(
            ratio=self.args.densify_ratio,
            iter=self.args.densify_iter,
            node_source='known',
            time_source='eval',
            plot=self.args.plot_densify,
            densify_unknown=self.args.densify_unknown,
            densify_uniform=self.args.densify_uniform
        )
        yaml.dump(self.interp_params, open(outdir(f'{args.workdir}/interp/_interp_params.yaml'), 'w'))

        known_coords = [i for i in range(A_first.shape[0]) if i not in self.loader.densified_set]
        X_predict = self._get_model_output(X_batch_groups, A_first, A_sub, densified_coords)

        # if t_sr emabled, X_predict will be longer than the input, with the same duration as Y_batch_groups
        # Put back input values from original sensors (only let the model predict the densified ones)
        X_original = rearrange(Y_batch_groups, 'g b 1 n p -> n (g b p)')
        X_predict[known_coords, :] = X_original[known_coords, :X_predict.shape[1]]

        X_predict = self.loader.scaler.inv(X_predict)

        self.densified_coords = densified_coords
        self.dates = dates
        self.X_predict = X_predict

        _t_range = t_range if isinstance(t_range, list) else [t_range, t_range + 1, 1]
        runner_params = [{
            't': t,
            'tsr_id': tsr_id
        } for t in range(*_t_range) for tsr_id in range(self.train_args.t_sr)]

        # Only multiprocessing backend works. not sure why
        results = Parallel(backend='multiprocessing', n_jobs=NUM_PROCESS)(delayed(self._interp_worker)(
            t=rp['t'],
            tsr_id=rp['tsr_id'],
            plot_bg=True,
            plot_scatter=True,
        ) for rp in runner_params)

        # store interp results for client usage
        self.interp_results = deepcopy(results)

        results = pd.DataFrame(results)
        results.to_csv(f'{args.workdir}/interp/_stat_{datestr()[-4:]}.csv', index=False)

        plt.figure(figsize=(12, 3))
        plt.grid(True)
        # plot mse_masked and mse_densified
        plt.plot(results['t'], results['mse_masked'], label=f'mse_masked, avg = {results["mse_masked"].mean()}')
        plt.plot(results['t'], results['mse_densified'], label=f'mse_densified = {results["mse_densified"].mean()}')
        plt.legend()
        plt.savefig(f'{args.workdir}/interp/_stat_{datestr()[-4:]}.pdf', dpi=300)


        plt.figure(figsize=(12, 3))
        plt.grid(True)
        # plot mse_masked and mse_densified
        plt.plot(results['t'], results['ssim_masked'], label=f'ssim_masked, avg = {results["ssim_masked"].mean()}')
        plt.plot(results['t'], results['ssim_densified'], label=f'ssim_densified = {results["ssim_densified"].mean()}')
        plt.legend()
        plt.savefig(f'{args.workdir}/interp/_stat_ssim_{datestr()[-4:]}.pdf', dpi=300)

    def _imputation_worker(self):
        pass

    def imputation(
            self,
            plot_t: Union[Iterable[int], int] = None,  # [start, end, step] or int
    ):
        """Calculate imputation loss. Just like the eval function in the trainer.
        """
        X_batch_groups, Y_batch_groups, A_first, A_sub, coords = self.loader.sample_eval()
        X_predict = self._get_model_output(X_batch_groups, A_first, A_sub, coords)

        X_predict = self.loader.scaler.inv(X_predict)

        Y_predict = rearrange(Y_batch_groups, 'g b 1 n p -> n (g b p)')
        Y_predict = self.loader.scaler.inv(Y_predict)

        # Only calculate rmse for unknown nodes
        loss_flag = np.zeros_like(Y_predict)
        loss_flag[list(self.loader.unknown_set), :] = 1

        input_flag = np.ones_like(Y_predict)

        truth = Y_predict * loss_flag * input_flag
        pred = X_predict * loss_flag * input_flag
        rmse = mse(pred, truth, squared=False)
        mae = np.abs(pred - truth).mean()
        truth_non0 = truth != 0
        mape = np.abs((truth[truth_non0] - pred[truth_non0]) / truth[truth_non0]).mean()
        logger.info(f'Imputation RMSE: {rmse:.5f}, MAE: {mae:.5f}, MAPE: {mape:.5f}')

        if plot_t:
            _plot_t = plot_t if isinstance(plot_t, list) else [plot_t, plot_t + 1, 1]

    def calc_uncertainty(
        self,
        plot_t: Union[Iterable[int], int] = [0, 1, 1],  # [start, end, step]
        topk=5,
    ):
        """
        Quantify uncertainties of sensor measure values
        """
        X_group_0, A_first_0, A_sub_0, coords_0, masked_half_0, X_group_1, A_first_1, A_sub_1, coords_1, masked_half_1, X_dates = \
            self.loader.sample_uncertainty(node_source='known', time_source='eval')
        X_observed = self.loader.X_eval_all[list(self.loader.known_set), :]

        X_predict_0 = self._get_model_output(X_group_0, A_first_0, A_sub_0, coords_0)
        X_predict_1 = self._get_model_output(X_group_1, A_first_1, A_sub_1, coords_1)

        X_observed = self.loader.scaler.inv(X_observed)[:, :X_predict_0.shape[1]]
        X_predict_0 = self.loader.scaler.inv(X_predict_0)
        X_predict_1 = self.loader.scaler.inv(X_predict_1)
        X_predict_mix = X_predict_0.copy()
        X_predict_mix[list(masked_half_0), :] = X_predict_1[list(masked_half_0), :]

        # shape of [num_sensors, num_timesteps], predict relative to observed.
        #   For variance > 0, the arrow should point up wards.
        variance = X_observed - X_predict_mix

        A_all_distance = compute_A(coords_0, norm=False)
        A_top_distance = adj_top_k(A_all_distance, topk, largest=False)
        A_top_distance[A_top_distance < 1e-6] = np.nan
        avg_distance = bn.nanmean(A_top_distance, axis=1)

        files_to_save = [
            (variance, f'variance'),
            (avg_distance, f'avg_distance'),
        ]
        for data, name in files_to_save:
            npsave(data, outdir(f'{self.args.workdir}/vis/{name}.npy'), with_png=True)

        if plot_t:
            _t_range = plot_t if isinstance(plot_t, list) else [plot_t, plot_t + 1, 1]
            for t in range(*_t_range):
                logger.debug(f'Plotting uncertainty at t={t}...')
                figsize = (12, 6)
                fig, ax = plt.subplots(figsize=figsize)
                self.loader.terrain_gdf.plot(ax=ax, color='none', edgecolor='black', alpha=0.5)
                x_min, y_min, x_max, y_max = self.loader.terrain_gdf.total_bounds
                vdata = variance[:, t]
                greater = vdata > 0
                vmin, vmax = vdata.min(), vdata.max()
                sizes = np.power(np.abs(vdata) / np.max(vdata), 0.5) * 10

                ax.scatter(
                    coords_0[greater, 0],
                    coords_0[greater, 1],
                    cmap='Spectral_r',
                    c=vdata[greater],
                    s=sizes[greater],
                    vmin=vmin,
                    vmax=vmax,
                    marker='^',
                    linewidths=0.4,
                    edgecolors='white',
                    alpha=0.5,
                )
                ax.scatter(
                    coords_0[~greater, 0],
                    coords_0[~greater, 1],
                    c=vdata[~greater],
                    cmap='Spectral_r',
                    s=sizes[~greater],
                    marker='v',
                    vmin=vmin,
                    vmax=vmax,
                    linewidths=0.4,
                    edgecolors='white',
                    alpha=0.5,
                )
                fig.tight_layout()
                plt.title(f'Uncertainty at t={t}')
                plt.savefig(outdir(f'{args.workdir}/uncertainty/uncertainty_t{t}.pdf'), dpi=300)

    def gen_client_data(self):

        if not self.densified_coords is None:
            logger.warning(f'self.densified_coords not found. Run interpolation first.')
        x_min, y_min, x_max, y_max = self.loader.terrain_gdf.total_bounds
        meta = {
            'train_args': vars(self.train_args),
            'interp_method': self.interp_method,
            'interp_params': self.interp_params,
            'interp_results': self.interp_results,
            'coords': self.loader.coords.tolist(),
            'densified_coords': self.densified_coords.tolist(),
            'known_set': list(self.loader.known_set),
            'densified_set': list(self.loader.densified_set),
            'terrain': '/terrain.geojson',
            'terrain_bound': {
                'x_min': x_min,
                'y_min': y_min,
                'x_max': x_max,
                'y_max': y_max,
            },
            'value_predict': '/x_predict.npy',  # [ densified_coords.shape[0], len(dates) ]
            'value_truth': '/x_truth.npy',  # [ all_coords.shape[0], len(dates) ]
            'sensor_density': '/density.npy',
            'uncertainty_variance': '/variance.npy',
            'uncertainty_avg_distance': '/avg_distance.npy',
        }
        # save terrain.
        X_truth = self.loader.scaler.inv(self.loader.X_eval_all)
        self.loader.terrain_gdf.to_file(outdir(f'{args.workdir}/vis/terrain.geojson'), driver='GeoJSON')
        npsave(self.X_predict, outdir(f'{args.workdir}/vis/x_predict.npy'), with_png=True)
        npsave(X_truth, outdir(f'{args.workdir}/vis/x_truth.npy'), with_png=True)
        json.dump(meta, open(outdir(f'{args.workdir}/vis/meta.json'), 'w'), cls=NpEncoder)

        # save all coords.
        # json.dump(self.loader.coords.tolist(), open(outdir(f'{args.workdir}/vis/coords.json'), 'w'))
        # # save known set
        # json.dump(self.loader.known_set.tolist(), open(outdir(f'{args.workdir}/vis/known_set.json'), 'w'))
        # if self.densified_coords:
        #     json.dump(self.densified_coords.tolist(), open(outdir(f'{args.workdir}/vis/densified_coords.json'), 'w'))
        #     json.dump(list(self.loader.densified_set), open(outdir(f'{args.workdir}/vis/densified_set.json'), 'w'))
        # else:
        #     logger.warning(f'self.densified_coords not found. Run interpolation first.')
        # save dates
        # if self.dates:
        #     json.dump([d.strftime('%Y-%m-%d %H:%M:%S') for d in self.dates], open(outdir(f'{args.workdir}/vis/dates.json'), 'w'))
        # else:
        #     logger.warning(f'self.dates not found. Run interpolation first.')


# Test notes:

# ./results/imterp_2403081924_ushcn_k3_p16_z32_wt5_pe16_tsr_1 -> Used for generating density figures

if __name__ == '__main__':
    # Required for multiprocessing
    th.multiprocessing.set_start_method('spawn', force=True)

    parser = ArgumentParser()

    parser.add_argument(
        '--workdir',
        type=str,
        default='./results/imterp_2403081924_ushcn_k3_p16_z32_wt5_pe16_tsr_1',
        help='Directory containing the checkpoints and training params (generated by trainer).')

    parser.add_argument('--checkpoint', type=str, default=f'e2000', help='Prefix of the checkpoint.')
    parser.add_argument('--densify_ratio', type=float, default=0.4, help='Densify ratio.')
    parser.add_argument('--densify_iter', type=int, default=2, help='Num of iterations.')
    parser.add_argument('--plot', type=bool, default=False, help='If save plot results for interolation.')
    parser.add_argument('--plot_densify', type=bool, default=False, help='If save plots for densify.')
    parser.add_argument('--densify_unknown',
                        type=bool,
                        default=False,
                        help='If the densified coords are the unknown coords. Densify heatmap will not generate.')
    parser.add_argument('--densify_uniform',
                        type=bool,
                        default=False,
                        help='If use uniform PDF for densification')


    parser.add_argument('--grid_size', type=int, default=1000, help='Grid width & height for spatial interpolation')

    parser.add_argument('--action', type=str, default='interp', help='Action to perform. (interp, imputation)')
    parser.add_argument('--interp_trange', type=str, default='0,100,1', help='Temporal range for interpolation.')

    args = parser.parse_args()

    core = ImTerp(args)

    if args.action == 'interp':
        trange = list(map(int, args.interp_trange.split(',')))
        core.interp(t_range=trange)
        core.calc_uncertainty()
        core.gen_client_data()
    elif args.action == 'imputation':
        core.imputation()

    # core.imputation()
