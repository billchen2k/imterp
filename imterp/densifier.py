

import numpy as np
from geopandas import gpd
from matplotlib import pyplot as plt

from utils import np2png, npsave, outdir, region_mask, logger
from scipy import stats
from scipy.spatial import Voronoi, voronoi_plot_2d

class Densifier(object):

    def __init__(
        self,
        coords: np.ndarray,
        terrain_gdf: gpd.GeoDataFrame,
        grid_size: int = 1000,
        outdir: str = None,
    ) -> None:
        self.coords = coords
        self.terrain_gdf = terrain_gdf
        self.mask = region_mask(terrain_gdf, np.zeros((grid_size, grid_size)))
        self.outdir = outdir
        self.rng = np.random.RandomState(2024)

    def find_centroid(self, vertices: np.ndarray) -> np.ndarray:
        """
        Find the centroid of a Voroni region described by `vertices`,
        and return a np array with the x and y coords of that centroid.
        https://en.wikipedia.org/wiki/Centroid#Of_a_polygon

        Args:
            vertices (np.array): a numpy array with shape [n, 2]

        Returns:
            np.ndarray: a numpy array that defines the x, y coords of the centroid described by `vertices`
        """
        area = 0
        centroid_x = 0
        centroid_y = 0
        for i in range(len(vertices) - 1):
            step = (vertices[i  , 0] * vertices[i+1, 1]) - \
                   (vertices[i+1, 0] * vertices[i  , 1])
            area += step
            centroid_x += (vertices[i, 0] + vertices[i + 1, 0]) * step
            centroid_y += (vertices[i, 1] + vertices[i + 1, 1]) * step
        area /= 2
        if area == 0:
            area += 1e-7
        centroid_x = (1.0 / (6.0 * area)) * centroid_x
        centroid_y = (1.0 / (6.0 * area)) * centroid_y
        return np.array([centroid_x, centroid_y])

    def within_region_mask(self, coord: np.ndarray) -> bool:
        """
        Args:
            coord (np.ndarray): shape (2,): (lng, lat)

        Returns:
            bool: if a given coordinates is within in the region mask.
        """
        x_min, y_min, x_max, y_max = self.terrain_gdf.total_bounds
        mask = self.mask
        mask_w = int((coord[0] - x_min) / (x_max - x_min) * mask.shape[0])
        mask_h = int((coord[1] - y_min) / (y_max - y_min) * mask.shape[1])
        if mask_w > mask.shape[0] - 1 or mask_w < 0 or mask_h > mask.shape[1] - 1 or mask_h < 0:
            return False
        return True if mask[mask_h, mask_w] > 0 else False

    def densify(self, ratio: float = 0.3, iter: int = 1, plot: bool = False) -> np.ndarray:
        """Densify sensor coordinates.

        Args:
            ratio (float, optional): Densify Ratio. Defaults to 0.3.
            iter (int, optional): Max num of iterations for CVVT. Defaults to 1.
            plot (bool, optional): If save plot. Defaults to False.

        Returns:
            np.ndarray: The densified sensor coordinates of shape (n, 2),
              where n = self.coords.shape[0] * (1 + ratio)
            set: The set for the index of densified coordinates.
        """
        x_min, y_min, x_max, y_max = self.terrain_gdf.total_bounds
        x_grid, y_grid = np.mgrid[x_min:x_max:1000j, y_min:y_max:1000j]
        positions = np.vstack([x_grid.ravel(), y_grid.ravel()])

        logger.debug(f'KDE in progress...')
        kernel = stats.gaussian_kde(
            self.coords.T,
            bw_method='scott',
        )
        values = kernel(positions)
        dist = np.reshape(values, x_grid.shape)
        mask = region_mask(self.terrain_gdf, dist)

        dist = np.flip(np.rot90(dist), axis=0)
        dist_inv = (dist.max() - dist) * mask
        dist_inv = np.power(dist_inv, 1.5)

        if self.outdir:
            figsize = (12, 6)
            fig, ax = plt.subplots(figsize=figsize)
            ax.imshow(
                dist_inv,
                # cmap=plt.cm.Spectral_r,
                cmap=plt.cm.get_cmap('Blues'), # min = white, max = blue
                extent=[x_min, x_max, y_min, y_max],
                origin='lower',
            )
            plt.title('sampling density')
            plt.savefig(outdir(f'{self.outdir}/densify/sampling_densifty.pdf'))
            npsave(np.flip(np.where(mask == 0, np.nan, dist_inv), axis=0), outdir(f'{self.outdir}/vis/density.png'), with_png=True)

        num_densify = int(ratio * self.coords.shape[0])
        dist_inv_flat = dist_inv.ravel() / np.sum(dist_inv)
        # indices = np.random.choice(len(dist_inv_flat), size=num_densify, p=dist_inv_flat)
        indices = self.rng.choice(len(dist_inv_flat), size=num_densify, p=dist_inv_flat)
        # dist_inv = (dist.max() - dist) * mask

        # Convert indices to 2D coordinates
        y_idx, x_idx = np.unravel_index(indices, dist_inv.shape)
        x_coords = x_idx / dist_inv.shape[0] * (x_max - x_min) + x_min
        y_coords = y_idx / dist_inv.shape[1] * (y_max - y_min) + y_min
        densified_coords = np.vstack((np.vstack((x_coords, y_coords)).T))
        densified_set = set(range(self.coords.shape[0], self.coords.shape[0] + num_densify))

        skipped_coords = []
        # First sensor_coords, then densified_coords
        relaxed_coords = np.copy(densified_coords)

        for i in range(iter):
            logger.debug(f'Running CVT, iteration {i}...')
            skipped_coords = []
            vor = Voronoi(np.vstack((self.coords, relaxed_coords)), qhull_options='Qbb Qc Qx')
            for p_idx, reg_idx in enumerate(vor.point_region):  # point index & region index
                if not p_idx in densified_set:  # do not relax original sensors
                    continue
                # the region is a series of indices into self.voronoi.vertices
                # remove point at infinity, designated by index -1
                region = [i for i in vor.regions[reg_idx] if i != -1]
                # remove points outside the regional mask
                if any([not self.within_region_mask(vor.vertices[i]) for i in region]):
                    skipped_coords.append(vor.points[p_idx])
                    continue
                if len(region) < 3:
                    skipped_coords.append(vor.points[p_idx])
                    continue
                # enclose the polygon
                region = region + [region[0]]
                centroids = self.find_centroid(vor.vertices[region])
                relaxed_coords[p_idx - self.coords.shape[0]] = centroids
            skipped_coords = np.array(skipped_coords)
            if plot and self.outdir:

                def plot_single(original: bool, densified: bool, relaxed: bool, skipped: bool, plot_vor: bool = True):
                    identifiers = ['o', 'd', 'r', 's', 'v']
                    shown = [original, densified, relaxed, skipped, plot_vor]
                    suffix = ''.join([identifiers[i] for i in range(5) if shown[i]])
                    logger.debug(f'Making densification plot {suffix}...')
                    figsize = (12, 6)
                    fig, ax = plt.subplots(figsize=figsize)
                    self.terrain_gdf.plot(ax=ax, color='none', edgecolor='black', alpha=0.1)
                    if plot_vor:
                        voronoi_plot_2d(vor,
                                        point_size=0,
                                        show_vertices=False,
                                        ax=ax,
                                        line_width=0.5,
                                        line_alpha=0.8)
                    if original:
                        ax.scatter(self.coords[:, 0], self.coords[:, 1], color='royalblue', s=2, label='Original')
                    if densified:
                        ax.scatter(densified_coords[:, 0],
                                   densified_coords[:, 1],
                                   color='orangered',
                                   marker='^',
                                   s=3,
                                   alpha=0.3,
                                   label='Densified')
                    if relaxed:
                        ax.scatter(relaxed_coords[:, 0],
                                   relaxed_coords[:, 1],
                                   color='orangered',
                                   marker='^',
                                   s=3,
                                   label='Densified + Relaxed')
                    if skipped:
                        if skipped_coords.shape[0] > 0:
                            ax.scatter(skipped_coords[:, 0],
                                       skipped_coords[:, 1],
                                       color='purple',
                                       marker='^',
                                       s=2,
                                       label='Densified, Skipped Relaxing')
                        plt.legend()
                        plt.legend(loc='upper left', bbox_to_anchor=(1.04, 1)) # Put legend outside
                        plt.title(f'CVT, num_original={self.coords.shape[0]}, densify_ratio={ratio}')
                        plt.tight_layout()
                        plt.savefig(
                            outdir(f'./{self.outdir}/densify/iter{i}_{self.coords.shape[0]}_d{ratio}_{suffix}.pdf'))
                # Original, no d
                plot_single(True, False, False, False, False)
                # Original + Densification
                plot_single(True, True, True, True, True)
                # Original + Densification + Relaxed
                plot_single(True, False, True, True, True)
                # Densification + Relaxed
                # plot_single(False, True, True, False, False)
                # Relaxed
                plot_single(True, False, True, True, False)

        logger.debug(f'Densification done. ratio={ratio}, num_densified={relaxed_coords.shape[0]}')
        all_coords = np.concatenate((self.coords, relaxed_coords), axis=0)
        return all_coords, densified_set
