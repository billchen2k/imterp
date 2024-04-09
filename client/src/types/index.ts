export type IMapType = 'masked' | 'densified' | 'truth';

export interface IVisMeta {
  train_args: Partial<ITrainArgs>;
  interp_method: string;
  interp_params: Partial<IInterpParams>;
  interp_results: IInterpResult[];
  coords: number[][];
  densified_coords: number[][];
  known_set: number[];
  densified_set: number[];
  terrain: string;
  terrain_bound: ITerrainBound;
  value_predict: string;
  value_truth: string;
  sensor_density: string;
  uncertainty_variance: string;
  uncertainty_avg_distance: string;
}

export interface IInterpParams {
  neighbors: number;
  smoothing: number;
  epsilon: number;
  kernel: string;
  [key: string]: any;
}

export interface IInterpResult {
  t: number;
  date: string;
  mse_masked: number;
  mse_densified: number;
  ssim_masked?: number;
  ssim_densified?: number;
  diff: number;
  map_masked: string;
  map_densified: string;
  map_truth: string;
  vrange: number[];
}

export interface ITerrainBound {
  x_min: number;
  x_max: number;
  y_min: number;
  y_max: number;
}


export interface ITrainArgs {
  batch_size: number;
  config: string;
  dataset: string;
  dropout: number;
  epoch: number;
  ignore0: boolean;
  k: number;
  lr: number;
  masked_rate: number;
  max_nodes: number;
  note: string;
  p: number;
  pe: number;
  pe_scales: number;
  spec: string;
  t_sr: number;
  train_rate: number;
  unknown_rate: number;
  wt: number;
  z: number;
  [key: string]: any;
}
