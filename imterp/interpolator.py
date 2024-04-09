
from typing import Dict, Iterable, List

import bottleneck as bn
import numpy as np
from geopandas import gpd
from matplotlib import pyplot as plt
from matplotlib import tri
from scipy import interpolate, stats
from scipy.interpolate import (CloughTocher2DInterpolator, RBFInterpolator)

from utils import logger, region_mask


class Interpolator(object):

    def __init__(self,
                 sensor_coords: np.ndarray,
                 values: np.ndarray,
                 terrain_gdf: gpd.GeoDataFrame,
                 grid_size: int = 1000,
                 params: Dict = {},
                 outdir: str=None
                 ) -> None:
        """Interpolate data into a grid.

        Args:
            sensor_coords (np.ndarray): Sensor coordilates of shape [ num_sensors, 2 ] in the form of (lng, lat)
            values (np.ndarray): Sensor values of shape [ num_sensors, ]
            terrain_path (str): The path to the terrain shape file (.shp)
            grid_size (int, optional): Grid size. Defaults to 1000.
        """
        self.sensor_coords = sensor_coords
        self.values = values
        self.terrain_gdf = terrain_gdf
        self.mask = region_mask(self.terrain_gdf, np.zeros((grid_size, grid_size)))
        self.grid_size = grid_size
        self.x_min, self.y_min, self.x_max, self.y_max = self.terrain_gdf.total_bounds
        self.grid_x, self.grid_y = np.mgrid[self.x_min:self.x_max:self.grid_size * 1j,
                                            self.y_min:self.y_max:self.grid_size * 1j]
        self.params = params
        self.outdir = outdir


    def triangular(self):
        triang = tri.Triangulation(self.sensor_coords[:, 0], self.sensor_coords[:, 1])
        interpolator = tri.LinearTriInterpolator(triang, self.values)
        z = interpolator(self.grid_x.T, self.grid_y.T)
        z[self.mask == 0 & ~np.isnan(z)] = np.nan

        return z

    def cubic(self):
        interp = CloughTocher2DInterpolator(self.sensor_coords, self.values, tol=1e-4)
        z = interp(self.grid_x.T, self.grid_y.T)
        z[self.mask == 0 & ~np.isnan(z)] = np.nan
        return z

    def rbf(self):
        """
        Radial Basic Function Interpolation (from scipy)
        """
        interp = RBFInterpolator(
            self.sensor_coords,
            self.values,  #  neighbors=10,
            neighbors=self.params.get('neighbors') or 10,
            smoothing=self.params.get('smoothing') or 0,
            kernel=self.params.get('kernel') or 'gaussian',
            epsilon=self.params.get('epsilon') or 1,
        )
        x = self.grid_x.T.reshape(-1, 1)
        y = self.grid_y.T.reshape(-1, 1)
        xy = np.concatenate([x, y], axis=1)
        z = interp(xy).reshape(self.grid_size, self.grid_size)
        z[self.mask == 0 & ~np.isnan(z)] = np.nan
        return z

    def plot_result(
        self,
        known_coords: Iterable[int], # The list of index of known coordinates within sensor_coords
        vrange: Iterable[int],
        outdir: str,
        title: str = '',
        plot_bg: bool = True,
        plot_scatter: bool = True,
    ):
        logger.debug(f'Plotting interpolation result: {title}...')
        if self.interp_data is None:
            logger.warning('No interpolation data found. Please run interpolate() first.')
            return

        unknown_coords = [i for i in range(self.sensor_coords.shape[0]) if i not in known_coords]

        figsize = (12, 6)
        fig, ax = plt.subplots(figsize=figsize)
        self.terrain_gdf.plot(ax=ax, color='none', edgecolor='black', alpha=0.5)
        x_min, y_min, x_max, y_max = self.terrain_gdf.total_bounds
        vmin, vmax = vrange
        norm = plt.Normalize(vmin=vmin, vmax=vmax)
        cmap = plt.cm.Spectral_r
        if plot_bg:
            ax.imshow(
                self.interp_data,
                extent=(x_min, x_max, y_min, y_max),
                origin='lower',
                cmap='Spectral_r',
                vmin=vmin,
                vmax=vmax,
            )
        # axs[i].triplot(triang, color='red', alpha=0.01, linewidth=0.5)
        if plot_scatter:
            ax.scatter(
                self.sensor_coords[known_coords, 0],
                self.sensor_coords[known_coords, 1],
                cmap=plt.cm.Spectral_r,
                vmin=vmin,
                vmax=vmax,
                marker='o',
                s=9,
                linewidths=0.4,
                edgecolors='white',
                alpha=1,
                c=self.values[known_coords],
                label='known',
            )
            if len(unknown_coords) > 0:
                ax.scatter(
                    self.sensor_coords[unknown_coords, 0],
                    self.sensor_coords[unknown_coords, 1],
                    marker='X',
                    s=18,
                    linewidths=0.4,
                    edgecolors='white',
                    cmap=plt.cm.Spectral_r,
                    vmin=vmin,
                    vmax=vmax,
                    alpha=1,
                    c=self.values[unknown_coords],
                    label='unknown',
                )
            # for idx in range(self.sensor_coords.shape[0]):
            #     color = cmap(norm(self.values[idx]))
            #     marker = 'X' if (idx in unknown_coords) else 'o'
            #     ax.scatter(
            #         self.sensor_coords[idx, 0],
            #         self.sensor_coords[idx, 1],
            #         color=color,
            #         marker=marker,
            #         s=9 if marker == 'o' else 18,
            #         linewidths=0.4,
            #         edgecolors='white',
            #         alpha=1,
            #     )
        plt.legend(ncol=3, loc='upper center', bbox_to_anchor=(0.5, -0.05)) # Put legend outside
        ax.set_title(title)
        plt.tight_layout()
        plt.savefig(outdir)
        plt.clf()
        plt.close()


    def interp(self, method: str, savedir: str=None):
        """Interpolate data.

        Args:
            method (str): Interpolation method string

        Returns:
            np.ndarray: Shape of [grid_size, grid_size]
        """
        func_map = {
            'triangular': self.triangular,
            'cubic': self.cubic,
            'rbf': self.rbf,
        }
        if not method in func_map:
            logger.error(f'Unknown interpolation method {method}. Available: {list(func_map.keys())}')
        else:
            self.interp_data = func_map[method]()
            logger.info(
                f'Interpolation done ({method}). data range = [{bn.nanmin(self.values):.5f}, {bn.nanmax(self.values):.5f}]. z range = [{bn.nanmin(self.interp_data):.5f}, {bn.nanmax(self.interp_data):.5f}].'
            )
            return self.interp_data
