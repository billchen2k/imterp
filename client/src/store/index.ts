/*
 * index.ts
 * Project: imterp-client
 * Created: 2024-03-12 23:12:15
 * Author: Bill Chen (bill.chen@live.com)
 * -----
 * Last Modified: 2024-03-24 23:23:56
 * Modified By: Bill Chen (bill.chen@live.com)
 */

import {config} from '@/config';
import {create} from 'zustand';
import {combine, persist} from 'zustand/middleware';
import {immer} from 'zustand/middleware/immer';

export const initState = {
  workdir: config.workdirs[0],
  mapWidth: 600,
  gridSize: 50,
  scatterScale: 1,
  layerAlpha: {
    scatterOriginal: 1,
    scatterMore: 1,
    heat: 1,
    glyph: 1,
    hatch: 1,
    tiles: 1,
    terrain: 1,
    sx: 1,
    sy: 1,
  },
  glyph: {
    mode: 'grid' as 'grid' | 'item',
    scale: 1,
    horizontalScale: 1,
    verticalScale: 1,
    relativity: 0.8,
    interpInfluence: 1,
    interpUncertainty: 1,
  },
  hatch: {
    displayThreshold: 1,
    density: 1,
    brightness: 1,
    width: 1,
  },
  colormap: {
    name: 'Spectral',
    reverse: true,
    autoRange: true,
    range: [0, 100] as number[] | undefined,
  },
  lastRender: new Date().getTime(),
};

export const useMainStore = create(
    immer(
        persist(
            combine(initState,
                (set, get) => ({
                  setWorkdir: (workdir: string) => set({workdir}),
                  resetAll: () => set(initState, true),
                })),
            {
              name: 'imterp-client',
            }),
    ),

);

export const setMainState = useMainStore.setState;

export type MainState = ReturnType<typeof useMainStore.getState>;
