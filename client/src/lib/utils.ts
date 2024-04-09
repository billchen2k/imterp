
import {useMainStore} from '@/store';
import ndarray from 'ndarray';
import npyjs from 'npyjs';

/**
 * @param url The url of the npy file.
 * @returns The loaded npy file of ndarray type.
 */
export async function loadNpy(url: string): Promise<ndarray.NdArray> {
  const n = new npyjs();
  const arr = await n.load(url);
  const npyName = url.split('/').splice(-1)[0];
  console.log(`Loaded npy ${npyName}. Shape: ${arr.shape}, dtype: ${arr.dtype}`);
  // @ts-ignorea
  return ndarray(arr.data, arr.shape);
}

/**
 * Get url for fetching data from server.
 * The workdir will be automatically added to the path.
 */
export const getVisUrl = (path: string) => {
  const workdir = useMainStore.getState().workdir;
  return `/api/res/${workdir}/vis/${path}`;
};

/**
 * @param coords Coordinate of shape [n, 2]
 * @param indices Index set.
 * @returns Filtered coordinates with indices in coord array
 */
export const filteredCoords = (coords: number[][], indices: number[]): {idx: number, coord: number[]}[] => {
  const filtered = [];
  for (let i = 0; i < indices.length; i++) {
    filtered.push({idx: indices[i], coord: coords[indices[i]]});
  }
  return filtered;
};
