import Vis from '@/components/vis';
import {getVisUrl} from '@/lib/utils';
import {useMainStore} from '@/store';
import {IVisMeta} from '@/types';
import {Button, MenuItem, Select} from '@mui/material';
import useAxios from 'axios-hooks';
import * as React from 'react';

export interface IMapViewProps {
}

export function MapView(props: IMapViewProps) {
  const mainState = useMainStore((state) => state);
  const [currentT, setCurrentT] = React.useState(0);

  const [{data, loading, error}, axFetch] = useAxios<IVisMeta>({
    url: getVisUrl('meta.json'),
  });

  console.log(data?.interp_results);

  if (!data?.interp_results) {
    console.log(data);
    return <div>bad meta</div>;
  }

  return (
    <div className={'w-full h-full p-4'}>

      {error &&
        <div className={'text-red-500'}>Failed loading {mainState.workdir}: {error.message}</div>
      }

      {data &&
        <div className={'flex w-full flex-col gap-1'} style={{lineBreak: 'anywhere'}}>

          <div className={'font-mono text-sm'}>
            Train args: {JSON.stringify(data.train_args)}
          </div>

          <div className={'font-mono text-sm'}>
            Interp params: {JSON.stringify(data.interp_params)} ({data.interp_method})
          </div>

          <div className={'w-full flex flex-row gap-2 items-center'}>
            <div className={'text-sm'}>Select Time:</div>
            <Select className={'w-96'} value={currentT}
              onChange={(e) => setCurrentT(e.target.value as number)}
            >
              {data.interp_results.map((one, index) => {
                return <MenuItem value={index}>
                  {one.t}. {one.date}&nbsp;
                  <span className={one.mse_masked > one.mse_densified ? 'text-green-600' : 'text-yellow-600'}>
                  M {one.mse_masked.toFixed(4)} D {one.mse_densified.toFixed(4)}
                  </span>
                </MenuItem>;
              })}
            </Select>
            <Button onClick={() => setCurrentT(Math.max(0, currentT - 1))}>Last</Button>
            <Button onClick={() => setCurrentT(Math.min(data.interp_results.length - 1, currentT + 1))}>Next</Button>
          </div>

          <div className={'flex flex-row flex-wrap gap-3'}>
            <Vis title={'Masked'} meta={data} t={currentT} type={'masked'} />
            <Vis title={'Densified'} meta={data} t={currentT} type={'densified'} />
            <Vis title={'Truth'} meta={data} t={currentT} type={'truth'} />
          </div>
        </div>
      }
    </div>
  );
}
