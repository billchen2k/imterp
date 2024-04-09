import {config} from '@/config';
import {initState, setMainState, useMainStore} from '@/store';
import {Button, Checkbox, Divider, FormControl, FormControlLabel, InputLabel, MenuItem, Select, Slider, TextField} from '@mui/material';
import useAxios from 'axios-hooks';
import {useState, useRef, useEffect} from 'react';

export interface ISidebarProps {

}

export function Sidebar(props: ISidebarProps) {
  const [dataInfo, setDataInfo] = useState<string>('');

  const state = useMainStore((state) => state);
  const [value, setValue] = useState(JSON.stringify(state));

  const [axWorkdirs, axRefetch] = useAxios('/api/workdirs');

  const workdirs = [
    ...config.workdirs,
    ...(axWorkdirs.data ? axWorkdirs.data : []),
  ];

  let setStateByLoad = false;
  useEffect(()=>{
    console.log('change');
    if (!setStateByLoad) {
    // @ts-ignore
      setValue(JSON.stringify(state));
    } else setStateByLoad = false;
  }, [state, setStateByLoad]);

  // const setMainState = useCallback(_.throttle((...args: Parameters<typeof setMainState> ) => {
  //   setMainState(...args);
  // }, 1), [setMainState]);


  //   console.log('load mat', mat);
  //   setDataInfo(`shape: ${mat.shape}, dtype: ${mat.dtype}, 500, 500 = ${mat.get(500, 600)}`);
  // });

  //   <FormControl>
  //   <InputLabel id={'workdir-select-label'}>glyph.mode</InputLabel>
  //   <Select
  //     labelId={'workdir-select-label'}
  //     variant={'standard'}
  //     onChange={(e) => {
  //       setMainState((state) => {
  //         state.glyph.mode = e.target.value as typeof state.glyph.mode;
  //       });
  //     }}
  //     value={state.glyph.mode}
  //   >
  //     <MenuItem value={'item'}>item</MenuItem>
  //     <MenuItem value={'grid'}>grid</MenuItem>
  //   </Select>
  // </FormControl>

  return (
    <div className={'flex w-full h-full border-r flex-col gap-2 bg-gray-50 p-4 overflow-y-auto overflow-x-hidden'}>
      <div className={`text-2xl font-bold bg-bl left-0 pl-4 -translate-x-4 w-fit
         text-white bg-[#333333]
         py-1 pr-2`}>ImTerp</div>
      <div className={'text-base text-gray-800'}>
        Imputative Spatial Interpolation
      </div>
      <Divider />
      <FormControl variant={'standard'} size={'small'}>
        <InputLabel id={'workdir-select-label'}>Work Dir Select</InputLabel>
        <Select
          labelId={'workdir-select-label'}
          onChange={(e) => {
            setMainState({workdir: e.target.value as string});
            // flushInput();
          }}
          value={state.workdir}
        >
          {workdirs.map((one) => (
            <MenuItem value={one}>{one}</MenuItem>
          ))}
        </Select>
      </FormControl>
      <TextField label={'Current Working Dir'}
        value={state.workdir}
        onChange={(e) => {
          setMainState({workdir: e.target.value});
          // flushInput();
        }}
      />
      <div className={'grid grid-cols-2 gap-x-2 gap-y-1 items-center'}>
        <TextField label={'Map Width (in px)'} fullWidth
          value={state.mapWidth}
          type={'number'}
          onChange={(e) => {
            setMainState({mapWidth: parseInt(e.target.value) || 0});
            // flushInput();
          }}
        />
        <TextField label={'Grid Size (in px)'} fullWidth
          value={state.gridSize}
          type={'number'}
          onChange={(e) => {
            setMainState({gridSize: parseInt(e.target.value) || 0});
            // flushInput();
          }}
        />
        {(['scale', 'verticalScale', 'horizontalScale'] as Array<keyof typeof state.glyph>)
            .map((one) => (<>
              <TextField label={`glyph.${one}`} fullWidth
                value={state.glyph[one]}
                type={'number'}
                inputProps={{step: 0.05}}
                onChange={(e) => {
                  setMainState((state) => {
                    let v = parseFloat(e.target.value) || 0;
                    v = Math.max(-2, v);
                    v = Math.min(v, one === 'relativity'?1:3);
                    // @ts-ignore
                    state.glyph[one] = v;
                  });
                  // flushInput();
                }}
              />
            </>),
            )}

        <TextField label={`scatter.scale`} fullWidth
          value={state.scatterScale}
          type={'number'}
          inputProps={{step: 0.05}}
          onChange={(e) => {
            setMainState((state) => {
              const v = parseFloat(e.target.value) || 0;
              // @ts-ignore
              state.scatterScale = v;
            });
            // flushInput();
          }}
        />
        <>
          <FormControlLabel label={`glyph.relativity`} control={
            <Checkbox size='small'
              onChange={(e) => {
                if (e.target.checked) {
                  setMainState((state) => {
                    state.glyph['relativity'] = initState.glyph.relativity;
                  });
                } else {
                  setMainState((state) => {
                    state.glyph['relativity'] = 0;
                  });
                }
                // flushInput();
              }}
              checked={state.glyph['relativity'] >= initState.glyph['relativity']- 0.01} />
          } />
          <Slider
            onChange={(e, v) => {
              setMainState((state) => {
                state.glyph['relativity']= v as number;
              });
              // flushInput();
            }}
            size={'small'} min={0} max={1} step={0.01}
            value={state.glyph.relativity}
            valueLabelDisplay={'auto'}
          />
          <FormControlLabel label={`glyph.interpUncertainty`} control={
            <Checkbox size='small'
              onChange={(e) => {
                if (e.target.checked) {
                  setMainState((state) => {
                    state.glyph['interpUncertainty'] = initState.glyph.interpUncertainty;
                  });
                } else {
                  setMainState((state) => {
                    state.glyph['interpUncertainty'] = 0;
                  });
                }
                // flushInput();
              }}
              checked={state.glyph['interpUncertainty'] >= initState.glyph['interpUncertainty']- 0.01} />
          } />
          <Slider
            onChange={(e, v) => {
              setMainState((state) => {
                state.glyph['interpUncertainty']= v as number;
              });
              // flushInput();
            }}
            size={'small'} min={0} max={1} step={0.01}
            value={state.glyph.interpUncertainty}
            valueLabelDisplay={'auto'}
          />
          <FormControlLabel label={`glyph.interpInfluence`} control={
            <Checkbox size='small'
              onChange={(e) => {
                if (e.target.checked) {
                  setMainState((state) => {
                    state.glyph.interpInfluence = initState.glyph.interpInfluence;
                  });
                } else {
                  setMainState((state) => {
                    state.glyph.interpInfluence = -1;
                  });
                }
                // flushInput();
              }}
              checked={state.glyph.interpInfluence >= initState.glyph.interpInfluence - 0.01} />
          } />
          <Slider
            onChange={(e, v) => {
              setMainState((state) => {
                if (v != 0) {
                  state.glyph.interpInfluence = v as number;
                } else {
                  state.glyph.interpInfluence = 0.01;
                }
              });
              // flushInput();
            }}
            size={'small'} min={-1} max={1} step={0.01}
            value={state.glyph.interpInfluence}
            valueLabelDisplay={'auto'}
          />
        </>


        {(Object.keys(state.layerAlpha) as Array<keyof typeof state.layerAlpha>).map((one) => (<>
          <FormControlLabel label={`alpha.${one}`} control={
            <Checkbox size='small'
              onChange={(e) => {
                if (e.target.checked) {
                  setMainState((state) => {
                    state.layerAlpha[one] = initState.layerAlpha[one];
                  });
                } else {
                  setMainState((state) => {
                    state.layerAlpha[one] = 0;
                  });
                }
                // flushInput();
              }}
              checked={state.layerAlpha[one] >= initState.layerAlpha[one] - 0.01} />
          } />
          <Slider
            onChange={(e, v) => {
              setMainState((state) => {
                state.layerAlpha[one] = v as number;
              });
              // flushInput();
            }}
            size={'small'} min={0} max={1} step={0.01}
            value={state.layerAlpha[one]}
            valueLabelDisplay={'auto'}
          />
        </>
        ))}
        {(Object.keys(state.hatch) as Array<keyof typeof state.hatch>).map((one) => (<>
          <FormControlLabel label={`hatch.${one}`} control={
            <Checkbox size='small'
              onChange={(e) => {
                if (e.target.checked) {
                  setMainState((state) => {
                    state.hatch[one] = initState.hatch[one];
                  });
                } else {
                  setMainState((state) => {
                    state.hatch[one] = 0;
                  });
                }
                // flushInput();
              }}
              checked={state.hatch[one] >= initState.hatch[one] - 0.01} />
          } />
          <Slider
            onChange={(e, v) => {
              setMainState((state) => {
                state.hatch[one] = v as number;
              });
              // flushInput();
            }}
            size={'small'} min={0.1} max={1} step={0.01}
            value={state.hatch[one]}
            valueLabelDisplay={'auto'}
          />
        </>
        ))}

        <FormControlLabel label={'colormap.reverse'} control={
          <Checkbox size='small'
            checked={state.colormap.reverse}
            onChange={(e) => {
              setMainState((draft) => {
                draft.colormap.reverse = e.target.checked;
              });
              // flushInput();
            }}
          />} />

        <FormControl>
          <InputLabel id={'colormap-select-label'}>colormap</InputLabel>
          <Select
            labelId={'colormap-select-label'}
            onChange={(e) => {
              setMainState((state) => {
                state.colormap.name = e.target.value as string;
              });
              // flushInput();
            }}
            value={state.colormap.name}>
            {config.colormaps.map((one) =>
              <MenuItem value={one}>{one}</MenuItem>,
            )}
          </Select>
        </FormControl>

      </div>


      <Button variant={'outlined'}
        onClick={() => {
          state.resetAll();
        }}
      >Reset</Button>

      <Button variant={'contained'}
        onClick={() => {
          setMainState((state) => {
            state.lastRender = new Date().getTime();
          });
        }}
      >Render</Button>

      <Button variant={'outlined'}
        onClick={() => {
          setMainState((state) => {
            (Object.keys(state) as Array<keyof typeof state>).map((one)=>{
              // @ts-ignore
              const stateConfig = JSON.parse(value);
              // @ts-ignore
              state[one] = stateConfig[one];
            });
          });
          setStateByLoad = true;
        }}
      >Load</Button>
      <TextField
        label="config to load"
        value={value}
        onChange={(e)=>{
          setValue(e.target.value);
        }}
        variant="standard"
        multiline
      />
    </div>
  );
}
