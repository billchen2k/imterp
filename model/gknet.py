import joblib
import numpy as np
import torch
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from einops import einsum, rearrange, repeat

from data_loader import AuxInfo, GKDataLoader
from model.pna import AGGREGATORS, SCALERS
from utils import astensor, logger


class Align(nn.Module):

    def __init__(self, in_size, out_size):
        super().__init__()
        self.in_size = in_size
        self.out_size = out_size
        if in_size > out_size:
            self.conv1x1 = nn.Conv2d(in_size, out_size, 1)

    def forward(self, x: th.Tensor) -> th.Tensor:
        """Align feature size. If output size is smaller, apply 1x1 convolution;
           If output size is greater, apply padding to the feature dimension.

        Args:
            x (th.Tensor): Input tensor of shape [batch_size, in_size, num_nodes, num_timesteps]

        Returns:
            th.Tensor: Output tensor of shape [batch_size, out_size, num_nodes, num_timesteps]
        """
        if self.in_size > self.out_size:
            return self.conv1x1(x)
        if self.in_size < self.out_size:
            # Only pad the second dimension
            return F.pad(x, [0, 0, 0, self.out_size - self.in_size, 0, 0, 0, 0])

        return x


class TemporalConv(nn.Module):

    def __init__(
            self,
            in_size: int,
            out_size: int,
            kernel_size: int = 1,
            dilation: int = 1,
            act='linear',  # linear / sigmoid / glu
            dropout: float = 0.1) -> None:
        super().__init__()
        self.dilation = dilation
        self.dropout = dropout
        self.act = act
        self.out_size = out_size
        self.align = Align(in_size, out_size)
        if self.act == 'glu':
            self.conv = nn.Conv2d(in_size,
                                  out_size * 2,
                                  kernel_size=(1, kernel_size),
                                  dilation=dilation,
                                  padding='same')
        else:
            self.conv = nn.Conv2d(in_size, out_size, kernel_size=(1, kernel_size), dilation=dilation, padding='same')

    def forward(self, X: th.tensor) -> torch.Tensor:
        """Temporal convolution layer

        Args:
            X (th.tensor): Input X of shape [batch_size, in_size, num_nodes, num_timesteps]

        Returns:
            torch.Tensor: Output X of shape [batch_size, out_size, num_nodes, num_timesteps]
        """
        X_align = self.align(X)
        if self.act == 'linear':
            h = self.conv(X) + X_align
        elif self.act == 'sigmoid':
            h = torch.sigmoid(self.conv(X) + X_align)
        elif self.act == 'glu':
            out = self.conv(X)
            h = (out[:, :self.out_size, :, :] + X) * torch.sigmoid(out[:, self.out_size:, :, :])
        return F.dropout(h, self.dropout, training=self.training)


class SpatialConv(nn.Module):

    def __init__(self,
                 in_size: int,
                 out_size: int,
                 p: int,
                 info: AuxInfo,
                 aggregators=['softmin', 'softmax', 'normalized_mean', 'std', 'var', 'distance', 'd_std'],
                 scalers=['identity', 'amplification', 'attenuation'],
                 dropout: float = 0.1,
                 masking: bool = False) -> None:
        super().__init__()
        self.p = p
        self.in_size = in_size
        self.out_size = out_size
        self.dropout = dropout
        self.info = info
        self.aggregators = aggregators
        self.scalers = scalers
        self.masking = masking
        self.Theta = nn.Parameter(th.FloatTensor(
            len(aggregators) * len(scalers) * self.in_size,
            self.out_size,
        ))
        self.Bias = nn.Parameter(th.FloatTensor(self.out_size))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.Theta)
        stdv = 1. / np.sqrt(self.Bias.shape[0])
        nn.init.uniform_(self.Bias, -stdv, stdv)

    def forward(self, X: th.Tensor, A: th.Tensor) -> torch.Tensor:
        """PNA Convolution on the spatial domain.
        Temporal information is not considered here.

        Args:
            X (th.Tensor): Input tensor of shape [batch_size, in_size, num_nodes, num_timesteps]
            A (th.Tensor): Adjacency matrix of shape [num_nodes, num_nodes]

        Returns:
            torch.Tensor: Output tensor of shape [batch_size, out_size, num_nodes, num_timesteps]
        """
        # batch_size, in_size, num_nodes, num_timesteps = X.shape
        # src, dst = th.nonzero(A, as_tuple=True)
        # edge_feat = A[src, dst].reshape((-1, 1))
        # out_all = th.zeros((batch_size, self.out_size, num_nodes, num_timesteps)).to(X.device)
        # for batch in range(batch_size):
        #     graph = dgl.graph((src, dst))
        #     node_feat = rearrange(X[batch, :, :, :], 'c n p -> n (c p)')
        #     out = self.conv(graph, node_feat, edge_feat) # [num_nodes, out_size * p]
        #     out = rearrange(out, 'n (c p) -> c n p', c=self.out_size)
        #     out_all[batch] = out

        # b = batch, c = channel, n = num_nodes, p = timesteps
        input = rearrange(X, 'b c n p -> b p n c')
        out = th.cat(
            [AGGREGATORS[a](input, A) for a in self.aggregators],
            dim=3,
        )
        out = th.cat(
            [SCALERS[s](out, A, self.info.deg) for s in self.scalers],
            dim=3,
        )
        out = einsum(out, self.Theta, 'b p n c, c o -> b p n o') + self.Bias
        out = rearrange(out, 'b p n o -> b o n p')

        out = F.dropout(out, self.dropout, training=self.training)
        out = F.leaky_relu(out)
        return out

        # src, dst = th.nonzero(A, as_tuple=True)
        # bg = dgl.batch([dgl.graph((src, dst)) for _ in range(batch_size)])
        # print(bg.batch_num_nodes)
        # p = X.shape[3]
        # node_feat = X.permute((0, 2, 3, 1)).reshape((batch_size * num_nodes, -1))  # time x feature size fused
        # print('node_feat', node_feat.shape)
        # edge_feat = A[src, dst].reshape((-1, 1)).repeat(batch_size, 1)
        # print('edge feat', edge_feat.shape)
        # out = self.conv(bg, node_feat, edge_feat)  # batch, node_count, feature_size
        # out = out.reshape((batch_size, num_nodes, -1, p)).permute((0, 2, 1, 3))  # unpack time x feature size
        # out = F.dropout(out, self.dropout, training=self.training)
        return out


class PositionalEncoding(nn.Module):

    def __init__(
            self,
            info: AuxInfo,
            size_in: int = 9,
            size_pe: int = 16,
            num_scales: int = 16,
            device: str = th.device('cpu'),
            size_one_scale: int = 8,
    ):
        super().__init__()
        self.fnn = nn.Linear(num_scales * 2 * 2, size_pe)
        self.norm = nn.LayerNorm(size_pe)
        self.info = info
        self.num_scales = num_scales
        self.size_one_scale = size_one_scale
        self.size_pe = size_pe
        self.device = device
        self.freq_mat = self.calc_freq_mat()
        self.to(device)

    def calc_freq_mat(self):
        sigmas = th.exp(-th.arange(0, self.num_scales))
        sigma_max = 1
        sigma_min = sigmas.min()
        freq_mat = th.zeros((self.num_scales, 2)).to(self.device)
        for s in range(self.num_scales):
            freq_mat[s, :] = 1 / sigma_min * th.pow(sigma_max / sigma_min, s / (self.num_scales - 1))
        # Shape: [num_scales, 2]
        return freq_mat

    def to_delta(self, coords: th.Tensor):
        """Convert coordinates to delta coordinates.

        Args:
            coords (th.Tensor): Coordinates array of shape [num_nodes, 2]

        Returns:
            th.Tensor: Delta coordinates array of shape [num_nodes, 2]. Lng and lat are within range [0, 1]
        """
        delta_coords = th.Tensor(coords).to(self.device)
        delta_coords[:, 0] = (coords[:, 0] - self.info.min_lng) / (self.info.max_lng - self.info.min_lng)
        delta_coords[:, 1] = (coords[:, 1] - self.info.min_lat) / (self.info.max_lat - self.info.min_lat)
        return delta_coords

    def forward(self, X: th.Tensor, coords: th.Tensor) -> th.Tensor:
        """Get positional encoding of given X, concatenate it to X.

        Args:
            X (th.Tensor): Input X of shape [batch_size, in_size(1), num_nodes, num_timesteps]
            coords (th.Tensor): Coordinates array of shape [num_nodes, 2]

        Returns:
            th.Tensor: The X with Position Encoding Concatenated.
                Output X of shape [batch_size, in_size(1), num_nodes, num_timesteps + size_pe]
        """
        if self.size_pe <= 0:
            return X
        delta_coords = self.to_delta(coords=coords)

        # Each scale: cos & sin for lnt & lat
        pe = th.zeros((delta_coords.shape[0], self.num_scales * 2 * 2)).to(self.device)

        pe[:, 0::4] = th.cos(einsum(delta_coords, self.freq_mat, 'n lnglat, scale lnglat -> n scale'))
        pe[:, 1::4] = th.sin(einsum(delta_coords, self.freq_mat, 'n lnglat, scale lnglat -> n scale'))

        pe = self.fnn(pe)
        pe = self.norm(pe)

        pe = repeat(pe, 'n d -> b c n d', b=X.shape[0], c=X.shape[1])
        X_pe = th.cat((X, pe), dim=3)
        return X_pe


class GKNet(nn.Module):

    def __init__(
            self,
            info: AuxInfo,
            in_size: int = 1,
            out_size: int = 1,
            temporal_size: int = 9,
            temporal_sr: int = 1,
            hidden_size: int = 32,
            t_kernel_size: int = 3,
            pe_size: int = 0,
            t_dilation: int = 1,
            device: th.device = th.device('cpu'),
            dropout: float = 0.1,
            spec: str = 'STSTr',
    ):
        """GKNet Model

        Args:
            info (AuxInfo): AuxInfo object containing the min max of lng lat.
            in_size (int, optional): Input channel count. Defaults to 1.
            out_size (int, optional): Output channel count. Defaults to 1.
            temporal_size (int, optional): Temporal window size. Defaults to 9.
            temporal_sr (int, optional): Temporal super resolution rate. Defaults to 1.
            hidden_size (int, optional): Hidden space dimension. Defaults to 32.
            t_kernel_size (int, optional): Kernel size of temporal convolution. Defaults to 3.
            pe_size (int, optional): _description_. Defaults to 0.
            t_dilation (int, optional): _description_. Defaults to 1.
            device (th.device, optional): _description_. Defaults to th.device('cpu').
            dropout (float, optional): _description_. Defaults to 0.1.
            spec (str, optional): _description_. Defaults to 'STSTr'.
        """
        super().__init__()
        self.info = info
        self.pe = PositionalEncoding(info=info, size_pe=pe_size, device=device)
        self.s_conv0 = SpatialConv(
            in_size,
            hidden_size,
            info=self.info,
            p=temporal_size,
            dropout=dropout,
            scalers=['identity'],
            masking=True,
        )
        self.t_conv0 = TemporalConv(hidden_size,
                                    hidden_size,
                                    kernel_size=t_kernel_size,
                                    dilation=1,
                                    dropout=dropout,
                                    act='glu')
        self.layers = nn.ModuleList()
        self.layer_types = []
        for i, c in enumerate(spec[2:]):
            if c == 'S':
                self.layers.append(
                    SpatialConv(
                        hidden_size,
                        hidden_size,
                        info=self.info,
                        p=temporal_size,
                        dropout=dropout,
                        masking=True,
                    ))
                self.layer_types.append('S')
            if c == 'T':
                self.layers.append(
                    TemporalConv(hidden_size,
                                 hidden_size,
                                 kernel_size=t_kernel_size,
                                 dilation=t_dilation,
                                 act='sigmoid',
                                 dropout=dropout))
                self.layer_types.append('T')
        self.out_conv = nn.Sequential(
            nn.Conv2d(hidden_size, out_size, (1, 1), 1),
            nn.Linear(temporal_size + pe_size, temporal_size * temporal_sr),
        )
        if spec[-1] == 'r':
            self.out_act = nn.ReLU()
        elif spec[-1] == 't':
            self.out_act = nn.Tanh()
        elif spec[-1] == 's':
            self.out_act = nn.Sigmoid()
        elif spec[-1] == 'i':
            self.out_act = nn.Identity()
        elif spec[-1] == 'l':
            self.out_act = nn.LeakyReLU()
        self.device = device
        self.to(self.device)

        total_parameters = sum(p.numel() for p in self.parameters())
        logger.info(f'Total number of model parameters: {total_parameters}')

    def forward(
        self,
        X: th.tensor,
        A_first: th.Tensor,
        A_sub: th.Tensor,
        coords: th.Tensor,
        verbose: bool = False,
    ) -> th.Tensor:
        """Forward pass of the GKNet model.

        Args:
            X (th.tensor): Input X of shape [batch_size, in_size, num_nodes, num_timesteps]
            A_first (th.Tensor): Normalized adjacency matrix of the first spatial conv layer (masked): [num_nodes, num_nodes]
                A_first only allow messages to pass between known nodes.
            A_sub (th.Tensor): Normalized adjacency matrix of the subsequent spatial conv layers: [num_nodes, num_nodes]
                A_sub allow messages to pass between all nodes.
            coords (th.Tensor): Coordinates of the nodes of shape [num_nodes, 2] (Used for Positional Encoding)

        Returns:
            th.Tensor: Output X of shape [batch_size, out_size, num_nodes, num_timesteps]
        """
        X = self.pe(X, coords)
        X = self.s_conv0(X, A_first)
        verbose and print('X after s_conv0', X.shape)
        X = self.t_conv0(X)
        verbose and print('X after t_conv0', X.shape)
        for layer_type, layer in zip(self.layer_types, self.layers):
            if layer_type == 'S':
                X = layer(X, A_sub)
                verbose and print('X after S', X.shape)
            if layer_type == 'T':
                X = layer(X)
                verbose and print('X after T', X.shape)
        X = self.out_conv(X)
        if verbose:
            print('X after out_conv', X.shape)
        X = self.out_act(X)
        return X


if __name__ == '__main__':

    # Tensor shape test
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    p = 16
    batch_size = 16
    max_node = 1500
    loader = GKDataLoader(dataset='ushcn')
    model = GKNet(temporal_size=p, device=device, temporal_sr=1,
                  hidden_size=32, pe_size=16, info=loader.info, spec='STSTr')
    model.to('cuda')
    X = joblib.load('./data/ushcn/X.z')[:max_node, :]
    A = joblib.load('./data/ushcn/A.z')[:max_node, :max_node]
    A = th.tensor(A).float().to(device)
    for i in range(1):
        x = th.zeros(batch_size, 1, X.shape[0], p).float().to(device)
        for i in range(batch_size):
            x[i, 0, :, :] = th.tensor(X[:, i * p:(i + 1) * p])
        print(f'In shape: {x.shape}')
        out = model.forward(x, A, A, coords=astensor(loader.coords), verbose=True)
        print(out[0, :])
        print(f'GPU memory: {torch.cuda.memory_allocated() / (1024 ** 3)} GB')
        print(f'Out shape: {out.shape}, range: {th.min(out)}, {th.max(out)}')