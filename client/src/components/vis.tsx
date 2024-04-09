
import {filteredCoords, getVisUrl, loadNpy} from '@/lib/utils';
import {MainState, useMainStore} from '@/store';
import {IVisMeta} from '@/types';
import {Button} from '@mui/material';
import useAxios from 'axios-hooks';
import chroma from 'chroma-js';
import * as d3 from 'd3';
import * as d3Tile from 'd3-tile';
import {produce} from 'immer';
import {jsPDF} from 'jspdf';
import * as React from 'react';
import {useAsync} from 'react-async-hook';
import Map, {
  LngLatBounds,
  useMap,
} from 'react-map-gl';
import 'svg2pdf.js';

export interface IVisProps {
  title: string;
  meta: IVisMeta;
  t: number;
  type: 'masked' | 'densified' | 'truth';
}

export default function Vis(props: IVisProps) {
  const {meta, t, type} = props;
  const state = useMainStore((state) => state);
  const mapMeta = meta.interp_results[t];
  const inteprMapDataUrl = (
    type === 'masked' ? mapMeta.map_masked :
      type === 'densified' ? mapMeta.map_densified :
        type === 'truth' ? mapMeta.map_truth : ''
  );

  /**
   *  Use this local state if you want to rerender only when user clicks render.
   *  If you want it to update instantly with user input, use state.
   */
  const [localState, setLocalState] = React.useState<MainState>(state);

  const [terrain] = useAxios<GeoJSON.FeatureCollection>(getVisUrl(meta.terrain));

  /**
   *  Access these npy data using data.result. Don't add the data directly as useEffect dependency,
   *    use data.result instead.
   */
  const interpMapData = useAsync(loadNpy, [getVisUrl(inteprMapDataUrl)]);
  const valPredictData = useAsync(loadNpy, [getVisUrl(meta.value_predict)]);
  const valTruthData = useAsync(loadNpy, [getVisUrl(meta.value_truth)]);
  const avgDistanceData = useAsync(loadNpy, [getVisUrl(meta.uncertainty_avg_distance)]);
  const varianceData = useAsync(loadNpy, [getVisUrl(meta.uncertainty_variance)]);
  const sensorDensityData = useAsync(loadNpy, [getVisUrl(meta.sensor_density)]);

  const heatLayerRef = React.useRef<HTMLCanvasElement>(null);
  const hatchLayerRef = React.useRef<HTMLCanvasElement>(null);
  const scatterLayerRef = React.useRef<SVGSVGElement>(null);
  const glyphLayerRef = React.useRef<SVGSVGElement>(null);
  const tileLayerRef = React.useRef<SVGSVGElement>(null);
  const terrainLayerRef = React.useRef<SVGSVGElement>(null);

  const maps = useMap();

  const saveAsPDF = () => {
    if (!heatLayerRef.current) return;
    if (!hatchLayerRef.current) return;
    if (!scatterLayerRef.current) return;
    if (!terrainLayerRef.current) return;
    if (!glyphLayerRef.current) return;
    const width = localState.mapWidth;
    const height = getHeight();
    let doc = new jsPDF({
      putOnlyUsedFonts: true,
      unit: 'px',
      orientation: width > height ? 'l' : 'p',
      format: [width, height],
    });
    if (localState.layerAlpha.hatch > 0) {
      doc = doc.addImage(heatLayerRef.current, 'PNG', 0, 0, width, height);
    }
    if (localState.layerAlpha.hatch > 0) {
      doc = doc.addImage(hatchLayerRef.current, 'PNG', 0, 0, width, height);
    }
    const docPromises = [];
    if (localState.layerAlpha.scatterMore > 0 || localState.layerAlpha.scatterOriginal > 0) {
      docPromises.push(doc.svg(scatterLayerRef.current, {x: 0, y: 0, width: width, height: height, loadExternalStyleSheets: true}));
    }
    if (localState.layerAlpha.terrain > 0) {
      docPromises.push(doc.svg(terrainLayerRef.current, {x: 0, y: 0, width: width, height: height, loadExternalStyleSheets: true}));
    }
    if (localState.layerAlpha.glyph > 0) {
      docPromises.push(doc.svg(glyphLayerRef.current, {x: 0, y: 0, width: width, height: height, loadExternalStyleSheets: true}));
    }
    Promise.all(docPromises).then((res)=>{
      res[0].save(`${inteprMapDataUrl.split('.')[0]}.pdf`);
    });
    // docPro.then((newdoc)=>{
    //   newdoc.save(`${inteprMapDataUrl.split('.')[0]}.pdf`);
    // });
  };

  // Helper functions
  const getHeight = React.useCallback(() => {
    const {x_min, x_max, y_min, y_max} = meta.terrain_bound;
    return localState.mapWidth / (x_max - x_min) * (y_max - y_min);
  }, [localState.mapWidth, meta.terrain_bound]);

  const getMedian = React.useCallback((arr: number[]) => {
    if (arr.length == 1) return arr[0];
    else {
      const mid = Math.floor(arr.length / 2);
      if (arr.length % 2 == 0) return (arr[mid] + arr[mid - 1]) / 2;
      else return arr[mid];
    }
  }, []);

  const getSensorDensity = React.useCallback((posX: number, posY: number) => {
    if (!sensorDensityData.result) return NaN;
    const x = Math.floor(posX * sensorDensityData.result.shape[1]);
    const y = Math.floor(posY * sensorDensityData.result.shape[0]);
    // console.log(sensorDensityData.result.get(y, x), posY, posX, y, x);
    return sensorDensityData.result.get(y, x);
  }, [sensorDensityData.result]);

  const colormap = React.useMemo(() => {
    const {reverse, autoRange, range} = localState.colormap;
    let [vmin, vmax] = [mapMeta.vrange[0], mapMeta.vrange[1]];
    if (!autoRange) {
      [vmin, vmax] = range as number[];
    }
    // vmin = 0;
    // vmax= 100;
    const domain = reverse ? [vmax, vmin] : [vmin, vmax];
    return chroma.scale(localState.colormap.name)
        .domain(domain);
  }, [localState.colormap, mapMeta.vrange]);

  const projection = React.useMemo(() => {
    const {data, loading, error} = terrain;
    if (!data) return;
    const width = localState.mapWidth;
    const height = getHeight();
    const projection = d3.geoEquirectangular()
        .fitExtent([[0, 0], [width, height]], data);

    const map = maps[`map-${props.title}`];
    // map?.flyTo({
    //   // zoom: projection.scale(),
    //   center: projection.invert!([width / 2, height / 2]) || [45, 45],
    // });
    map?.fitBounds([
      projection.invert!([0, 0]) || [45, 45],
      projection.invert!([width, height]) || [45, 45],
    ], {
    });
    map?.resize();
    return projection;
  }, [localState.mapWidth, getHeight, terrain, maps, props.title]);

  // data for hatchLayer
  // interpolate sensorDensityData
  // return number[height,width], value in [1,1+upperbound](NaN = 0)
  const sensorDensityUpperBound = React.useMemo(() => {
    let sensorDensityUpperBound: number = 0;

    const [height, width] = [getHeight(), localState.mapWidth];
    if (sensorDensityData.result) {
      for (let i = 0; i < sensorDensityData.result.shape[0]; i++) {
        for (let j = 0; j < sensorDensityData.result.shape[1]; j++) {
          const val = isNaN(sensorDensityData.result.get(i, j)) ? 0 : sensorDensityData.result.get(i, j);
          sensorDensityUpperBound = Math.max(sensorDensityUpperBound, val);
        }
      }
    }
    return sensorDensityUpperBound;
  }, [sensorDensityData.result, localState.mapWidth, getHeight]);

  // data for glyphLayer
  // coord of grid center
  const gridCoords = React.useMemo(() => {
    const [width, height] = [localState.mapWidth, getHeight()];
    const gridCoords = [];
    for (let i = Math.floor(localState.gridSize / 2); i < height + localState.gridSize / 2; i += localState.gridSize) {
      for (let j = Math.floor(localState.gridSize / 2); j < width + localState.gridSize / 2; j += localState.gridSize) {
        gridCoords.push([j, i]);
      }
    }
    return gridCoords;
  }, [localState.mapWidth, getHeight, localState.gridSize]);

  // known sets grouped By Grid, value is Indices Of KnownSet
  const knownSetGroupsByGrid = React.useMemo(() => {
    const knownSetGroupsByGridArr: number[][] = Array.from({length: gridCoords.length}, () => []);
    if (knownSetGroupsByGridArr.length > 0 && projection !== undefined) {
      const gridsPerRow = Math.ceil(localState.mapWidth / localState.gridSize);
      meta.known_set.map((idxCoords, i) => {
        const point = projection(meta.coords[idxCoords] as [number, number]);
        if (point === null) return;
        const idxX = Math.floor(point[0] / localState.gridSize);
        const idxY = Math.floor(point[1] / localState.gridSize);
        // if((point[0]))
        knownSetGroupsByGridArr[idxY * gridsPerRow + idxX].push(i);
        return;
      });
    }
    return knownSetGroupsByGridArr;
  }, [projection, meta.known_set, meta.coords, gridCoords, localState.gridSize, localState.mapWidth]);

  // glyphData in each grid, return null if no point in a grid
  const glyphDataGrid = React.useMemo(() => {
    if (avgDistanceData.result && varianceData.result && gridCoords.length > 0) {
      // console.log('avgDistData', avgDistanceData.result!.data);
      const rawGlyphData = gridCoords.map((_, gridIndex) => {
        const knownSetGroup = knownSetGroupsByGrid[gridIndex];
        if (knownSetGroup.length == 0) return null;

        // there are NaNs in avgDist
        let validCnt = knownSetGroup.length;
        const interpolationUncertainty = knownSetGroup.map((d) => {
          if (isNaN(avgDistanceData.result!.get(d))) {
            console.log(avgDistanceData.result!.get(d), d);
            validCnt -= 1;
            return 0;
          }
          return avgDistanceData.result!.get(d);
        })
            .reduce((acc, cur) => acc + cur, 0) / ((0 == validCnt) ? 1 : validCnt);

        // measurementReliability
        const measurementReliability = knownSetGroup.map((d) => varianceData.result!.get(d, mapMeta.t))
            .reduce((acc, cur) => acc + cur, 0);

        // reliabilityVariance
        const knownValue = knownSetGroup.map((d) => varianceData.result!.get(d, mapMeta.t))
            .sort((a, b) => a - b);
        // knownValue.length should > 0

        const bottomQuartileVal = getMedian(knownValue.slice(0, Math.max(1, knownValue.length / 2)));
        const topQuartileVal = getMedian(knownValue.slice(knownValue.length / 2));
        // change avgVal to medianVal
        const avgVal = getMedian(knownValue);
        // const avgVal = knownValue.reduce((acc, cur)=>acc+cur, 0)/knownValue.length;
        return {
          interpolationUncertainty: interpolationUncertainty,
          measurementReliability: measurementReliability,
          bottomQuartileVal: bottomQuartileVal,
          topQuartileVal: topQuartileVal,
          avgVal: avgVal,
          minVal: knownValue[0],
          maxVal: knownValue[knownValue.length - 1],
        };
      });

      // console.log('rawGlyphDataGrid', rawGlyphData);

      // normalization
      const maxInterpolationUncertainty = Math.max(...rawGlyphData.map((d) => d ? d.interpolationUncertainty : NaN).filter((d) => !isNaN(d)));
      const minInterpolationUncertainty = Math.min(...rawGlyphData.map((d) => d ? d.interpolationUncertainty : NaN).filter((d) => !isNaN(d)));
      const minVal = Math.min(...rawGlyphData.map((d) => d ? d.minVal : NaN).filter((d) => !isNaN(d)));
      const maxVal = Math.max(...rawGlyphData.map((d) => d ? d.maxVal : NaN).filter((d) => !isNaN(d)));

      const interpInfluence = localState.glyph.interpInfluence;
      const scaleUncert = d3.scaleLinear().domain([minInterpolationUncertainty, maxInterpolationUncertainty])
          .range([0, 1]);
      const scaleWidth = d3.scaleLinear();
      if (interpInfluence < 0) {
        scaleWidth.domain([minInterpolationUncertainty, maxInterpolationUncertainty])
            .range([1+0.75*interpInfluence, 1]);
      } else {
        scaleWidth.domain([maxInterpolationUncertainty, minInterpolationUncertainty])
            .range([1-0.75*interpInfluence, 1]);
      }
      // .domain([maxInterpolationUncertainty, minInterpolationUncertainty])
      // .range(interpInfluence > 0 ? [1 - 0.5 * interpInfluence, 1] : [1, 1 - 0.5 * interpInfluence]);
      // .exponent(Math.abs(localState.glyph.interpInfluence));
      const scaleVal = d3.scaleLinear([0, Math.max(Math.abs(minVal), Math.abs(maxVal))], [0, 1]);
      const glyphDataArr = rawGlyphData.map((data) => {
        if (data === null) return null;
        return {
          glyphUncert: scaleUncert(data.interpolationUncertainty),
          glyphWidth: scaleWidth(data.interpolationUncertainty), // more Interpolation Uncertainty -> thiner
          glyphDirect: data.measurementReliability > 0 ? 1 : -1, // 1->upward
          glythAvgVal: data.avgVal < 0 ? -scaleVal(Math.abs(data.avgVal)) : scaleVal(data.avgVal),
          glythBottomQuartileVal: data.bottomQuartileVal < 0 ? -scaleVal(Math.abs(data.bottomQuartileVal)) : scaleVal(data.bottomQuartileVal),
          glythTopQuartileVal: data.topQuartileVal < 0 ? -scaleVal(Math.abs(data.topQuartileVal)) : scaleVal(data.topQuartileVal),
        };
      });
      return glyphDataArr;
    } else return [];
  }, [mapMeta.t, gridCoords, knownSetGroupsByGrid, avgDistanceData.result, varianceData.result, getMedian, localState.glyph.interpInfluence]);

  const glyphDataItem = React.useMemo(() => {
    if (avgDistanceData.result && varianceData.result) {
      const knownValue = meta.known_set.map((idx, knownSetIdx) => varianceData.result!.get(knownSetIdx, mapMeta.t)).sort((a, b) => a - b);
      const bottomQuartileVal = getMedian(knownValue.slice(0, Math.max(1, knownValue.length / 2)));
      const topQuartileVal = getMedian(knownValue.slice(knownValue.length / 2));

      const rawGlyphData = meta.known_set.map((idx, knownSetIdx) => {
        // there are NaNs in avgDist
        const interpolationUncertainty = isNaN(avgDistanceData.result!.get(knownSetIdx)) ? 0 : avgDistanceData.result!.get(knownSetIdx);

        // measurementReliability
        const measurementReliability = varianceData.result!.get(knownSetIdx, mapMeta.t);

        // reliabilityVariance
        // change avgVal to medianVal
        // const avgVal = knownValue.reduce((acc, cur)=>acc+cur, 0)/knownValue.length;
        // const avgVal = varianceData.result!.get(knownSetIdx, mapMeta.t);
        const avgVal = 0;
        return {
          interpolationUncertainty: interpolationUncertainty,
          measurementReliability: measurementReliability,
          avgVal: avgVal,
        };
      });

      // normalization
      const maxInterpolationUncertainty = Math.max(...rawGlyphData.map((d) => d ? d.interpolationUncertainty : NaN).filter((d) => !isNaN(d)));
      const minInterpolationUncertainty = Math.min(...rawGlyphData.map((d) => d ? d.interpolationUncertainty : NaN).filter((d) => !isNaN(d)));
      const minVal = knownValue[0];
      const maxVal = knownValue[knownValue.length - 1];

      const scaleWdith = d3.scaleLinear([maxInterpolationUncertainty, minInterpolationUncertainty], [0.0001, 0.8]);
      const scaleVal = d3.scaleLinear([0, Math.max(minVal, maxVal)], [0, 1]);
      const glythBottomQuartileVal = bottomQuartileVal < 0 ? -scaleVal(Math.abs(bottomQuartileVal)) : scaleVal(bottomQuartileVal);
      const glythTopQuartileVal = topQuartileVal < 0 ? -scaleVal(Math.abs(topQuartileVal)) : scaleVal(topQuartileVal);

      const glyphDataArr = rawGlyphData.map((data) => {
        if (data === null) return null;
        return {
          glyphUncert: scaleWdith(data.interpolationUncertainty), // to be fixed
          glyphWidth: scaleWdith(data.interpolationUncertainty), // more Interpolation Uncertainty -> thiner
          glyphDirect: data.measurementReliability > 0 ? 1 : -1, // 1->upward
          glythAvgVal: data.avgVal < 0 ? -scaleVal(Math.abs(data.avgVal)) : scaleVal(data.avgVal),
          glythBottomQuartileVal: glythBottomQuartileVal,
          glythTopQuartileVal: glythTopQuartileVal,
        };
      });
      return glyphDataArr;
    } else return [];
  }, [meta.known_set, mapMeta.t, avgDistanceData.result, varianceData.result, getMedian]);

  // data for scatterLayer
  const scatterData = React.useMemo(() => {
    console.log('compute scatterData...');
    if (valTruthData.result && valPredictData.result && projection) {
      const coordsOriginal = (
        type === 'masked' ? filteredCoords(meta.coords, meta.known_set) :
          type === 'densified' ? filteredCoords(meta.coords, meta.known_set) :
            type === 'truth' ? filteredCoords(meta.coords, meta.coords.map((d, i) => i)) : null
      );
      const coordsMore = (
        type === 'densified' ? filteredCoords(meta.densified_coords, meta.densified_set) : null
      );
      const ori = coordsOriginal?.map((d) => {
        return ({
          point: projection(d.coord as [number, number]),
          val: valTruthData.result!.get(d.idx, mapMeta.t),
        });
      });
      const more = coordsMore?.map((d) => ({
        point: projection(d.coord as [number, number]),
        val: valPredictData.result!.get(d.idx, mapMeta.t),
      }));
      return {
        originalData: ori?.map((d) => ({x: d.point ? d.point[0] : null, y: d.point ? d.point[1] : null, val: d.val})),
        moreData: more?.map((d) => ({x: d.point ? d.point[0] : null, y: d.point ? d.point[1] : null, val: d.val})),
      };
    } else return null;
  }, [valTruthData.result, valPredictData.result, projection, type, meta.coords, meta.known_set, meta.densified_coords, meta.densified_set, mapMeta.t]);


  /**
   * Listen to the render button.
   */
  React.useEffect(() => {
    if (state.lastRender !== localState.lastRender) {
      // Put the main state into local state (for rendering).
      //    `produce` is used for deep copying the object.
      console.log('rerendering ...');
      setLocalState(produce(state, (draft) => { }));
    }
  }, [state, localState.lastRender]);

  /**
   * tiles layer
   */
  React.useEffect(() => {
    if (!tileLayerRef.current) return;
    if (!projection) return;


    // @depreacted: using mapbox now.
    // return;

    console.log('Rendering tiles layer');
    const width = localState.mapWidth;
    const height = getHeight();
    // for debug
    const sx = localState.layerAlpha.sx;
    const sy = localState.layerAlpha.sy;

    const svg = d3.select(tileLayerRef.current)
        .attr('width', width)
        .attr('height', height)
        .attr('viewBox', [0, 0, width, height]);
    const image = svg.append('g')
        .attr('transform', `scale(${sx},${sy})`)
        .attr('pointer-events', 'none')
        .selectAll('image');
    // @ts-ignore
    {
      // @ts-ignore
      const url = (x, y, z) => `https://api.mapbox.com/styles/v1/mapbox/navigation-day-v1/tiles/${z}/${x}/${y}${devicePixelRatio > 1 ? '@2x' : ''}?access_token=pk.eyJ1IjoiYmlsbGNoZW4yayIsImEiOiJjbGxnbW1lY3gweXdzM2hvM2o4em14a2R4In0.tyOAJUUzv68K6g2SjYVw3Q`;
      // const url = (x, y, z) => `https://tile.openstreetmap.org/${z}/${x}/${y}.png`;
      // @ts-ignore
      const zoomed = function(transform) {
        console.log(transform);
        const tiles = tile(transform);
        console.log(tiles);
        // @ts-ignore
        image.data(tiles, (d) => d).join('image')
            // @ts-ignore
            .attr('xlink:href', (d) => url(...d3Tile.tileWrap(d)))
            // @ts-ignore
            .attr('x', ([x]) => (x + tiles.translate[0]) * tiles.scale)
            // @ts-ignore
            .attr('y', ([, y]) => (y + tiles.translate[1]) * tiles.scale)
            .attr('width', tiles.scale)
            .attr('height', tiles.scale);
      };

      const tile = d3Tile.tile()
          .extent([[0, 0], [width, height]])
          // .size([width, height])
          .tileSize(512)
          .clampX(false);

      const zoom = d3.zoom()
          .scaleExtent([1 << 8, 1 << 22])
          .extent([[0, 0], [width, height]])
          .on('zoom', ({transform}) => zoomed(transform));

      // @ts-ignore
      svg.call(zoom).call(zoom.transform, d3.zoomIdentity
          // .translate(-2329.4246866048893, 562.9028973152965)
          // .scale(3565.7751072609253));
          // @ts-ignore
          .translate(...projection([0, 0]))
          .scale(projection.scale() * 2 * Math.PI));
    }

    return () => {
      svg.selectAll('*').remove();
    };
  }, [projection, localState.mapWidth, getHeight, localState.layerAlpha]);

  /**
   * Terrain layer
   */
  React.useEffect(() => {
    const {data, loading, error} = terrain;
    if (!terrainLayerRef.current) return;
    if (!data) return;
    if (projection === undefined) return;

    console.log('Rendering terrain layer');
    const svg = d3.select(terrainLayerRef.current);

    const width = localState.mapWidth;
    const height = getHeight();

    svg.attr('width', width).attr('height', height);

    svg.selectAll('.terrain')
        .data(data.features)
        .enter().append('path')
        .attr('class', 'terrain')
        .attr('d', (d) =>
          d3.geoPath()
              .projection(projection)(d),
        )
        .attr('fill', 'none') // Set the fill, adjust as needed
        .attr('stroke', 'rgb(100,100,100)') // Set the stroke color for boundaries, adjust as needed
        .attr('stroke-width', 0.5); // Set the stroke width, adjust as needed

    return () => {
      svg.selectAll('*').remove();
    };
  }, [projection, terrain, localState.mapWidth, getHeight]);

  /**
   * Hatch layer
   */
  React.useEffect(() => {
    if (!hatchLayerRef.current) return;

    console.log('Rendering hatch layer');
    const start = performance.now();

    const canvas = d3.select(hatchLayerRef.current);
    const ctx = canvas.node()!.getContext('2d');
    if (!ctx) return;

    // 4x antialiasing
    const mult = 4;
    const [width, height] = [localState.mapWidth * mult, getHeight() * mult];
    canvas.attr('width', width).attr('height', height)
        .style('width', `${width / mult}px`)
        .style('height', `${height / mult}px`);

    // interpolate SensorDensity
    const sensorDensityRange = sensorDensityUpperBound - 0;

    const hatchDensityScale = d3.scaleLinear()
        .domain([0, 1])
        .range([width / 20, width / 120]);
    const hatchWidthScale = d3.scaleLinear()
        .domain([0, 1])
        .range([0, width / 400]);
    const hatchInterval = Math.floor(hatchDensityScale(localState.hatch.density));
    const hatchColor = d3.interpolate('black', 'white')(localState.hatch.brightness);
    const hatchRgb = d3.color(hatchColor)?.rgb();
    const getHatchAlpha = (v: number) => {
      return v < localState.hatch.displayThreshold? 0:Math.pow(v, 2) * 1.2;
    };

    ctx.lineWidth = hatchWidthScale(localState.hatch.width);


    for (let i = 0; i < height; i++) {
      for (let j = 0; j < width; j++) {
        const v = getSensorDensity(j / width, i / height) / sensorDensityRange;
        if (isNaN(v)) continue;
        ctx.fillStyle = `rgba(${hatchRgb?.r},${hatchRgb?.g},${hatchRgb?.b},${getHatchAlpha(v)})`;
        ctx.fillRect(j, i, 1, 1);
      }
    }

    ctx.globalCompositeOperation = 'destination-in';

    // removing this line will lead incorrect redering, why?
    ctx.beginPath();
    for (let i = 0; i < width + height; i += hatchInterval) {
      ctx.moveTo(i, 0);
      ctx.lineTo(i - height, height);
    }
    ctx.stroke();


    console.log(`Rendered hatch layer in ${performance.now() - start}ms`);
    return () => {
      ctx.clearRect(0, 0, width, height);
    };
  }, [sensorDensityUpperBound, getSensorDensity, localState.mapWidth, getHeight, localState.hatch]);

  /**
 * Map data layer
 */
  React.useEffect(() => {
    if (!heatLayerRef.current) return;
    if (!interpMapData.result) return;
    console.log('Rendering heat layer');
    const start = performance.now();
    const canvas = d3.select(heatLayerRef.current);
    const [width, height] = [localState.mapWidth, getHeight()];
    const mult = 2.5;

    canvas
        .attr('width', width * mult)
        .attr('height', height * mult)
        .style('width', `${width}px`)
        .style('height', `${height}px`);

    // interpMapData.result is a 2d array. Render it on the canvas pixel by pixel using colormap. Ignore NaN.
    const ctx = canvas.node()!.getContext('2d');
    if (!ctx) return;
    ctx.globalAlpha = 1;
    ctx.imageSmoothingEnabled = false;
    const [pixelW, pixelH] = [width * mult / interpMapData.result.shape[1], height * mult / interpMapData.result.shape[0]];
    const [marginW, marginH] = [pixelW / 3, pixelH / 3];
    for (let i = 0; i < interpMapData.result.shape[0]; i++) {
      for (let j = 0; j < interpMapData.result.shape[1]; j++) {
        const v = interpMapData.result.get(i, j);
        if (isNaN(v)) continue;
        const colorRgba = colormap(v).rgba();
        ctx.fillStyle = `rgba(${colorRgba[0]}, ${colorRgba[1]}, ${colorRgba[2]}, ${1})`;
        /**
         * Add margin to make the background opaque (Avoid antialiasing)
         * https://stackoverflow.com/questions/73637669/how-to-draw-opaque-lines-on-an-html-canvas
         */
        ctx.fillRect(j * pixelW - marginW, i * pixelH - marginW, pixelW + 2 * marginW, pixelH + 2 * marginH);
      }
    }
    console.log(`Rendered heat layer in ${performance.now() - start}ms`);
    return () => {
      ctx.clearRect(0, 0, width, height);
    };
  }, [interpMapData.result, localState.mapWidth, localState.layerAlpha.heat, colormap, getHeight]);

  /**
 * Scatter layer
 */
  React.useEffect(() => {
    if (!scatterData) return;

    console.log('Rendering scatter layer');
    const start = performance.now();

    const svg = d3.select(scatterLayerRef.current);
    const width = localState.mapWidth;
    const height = getHeight();
    const scatterSize = width/150 * localState.scatterScale;
    const strokeWidth = Math.max(width / 1500, 0.5) * localState.scatterScale;
    // `corssWidth` refer to the shoter edge, `crossHeight` refer to ther longer one
    const crossWidth = scatterSize / 3;
    const crossHeight = scatterSize;

    const symbolStroke = '#555555';
    svg.attr('width', width).attr('height', height);
    if (scatterData.originalData) {
      svg.append('g')
          .selectAll('.originalData')
          .data(scatterData.originalData)
          .enter().append('circle')
          .attr('cx', (d) => d.x)
          .attr('cy', (d) => d.y)
          .attr('r', scatterSize / 2)
          .attr('fill', (d, i) => {
            return colormap(d.val).css();
          })
          .attr('stroke', symbolStroke)
          .attr('stroke-width', strokeWidth)
          .attr('opacity', localState.layerAlpha.scatterOriginal);
    }

    if (scatterData.moreData) {
      const cross = svg.append('g')
          .selectAll('.moreData')
          .data(scatterData.moreData)
          .enter().append('g')
          // .attr('transform', (d) => `rotate(45, ${d.x}, ${d.y})`)
          .attr('transform', (d) => `translate(${d.x}, ${d.y})`)
          // .attr('fill', (d) => colormap(d.val).css())
          .attr('opacity', localState.layerAlpha.scatterMore);

      cross.append('path')
          .attr('d',
              d3.symbol()
                  .type(d3.symbolCross)
                  .size(scatterSize * 7)(),
          )
          .attr('stroke', symbolStroke)
          .attr('stroke-width', strokeWidth)
          .attr('fill', (d) => colormap(d.val).css());

      // // stroke
      // cross.append('rect')
      //     .attr('width', crossWidth + strokeWidth * 2)
      //     .attr('height', crossHeight + strokeWidth * 2)
      //     .attr('x', (d) => d.x ? d.x - crossWidth / 2 - strokeWidth : 0)
      //     .attr('y', (d) => d.y ? d.y - crossHeight / 2 - strokeWidth : 0)
      //     .attr('fill', 'white');
      // cross.append('rect')
      //     .attr('width', crossHeight + strokeWidth * 2)
      //     .attr('height', crossWidth + strokeWidth * 2)
      //     .attr('y', (d) => d.y ? d.y - crossWidth / 2 - strokeWidth : 0)
      //     .attr('x', (d) => d.x ? d.x - crossHeight / 2 - strokeWidth : 0)
      //     .attr('fill', 'white');
      // // fill
      // cross.append('rect')
      //     .attr('width', crossWidth)
      //     .attr('height', crossHeight)
      //     .attr('x', (d) => d.x ? d.x - crossWidth / 2 : 0)
      //     .attr('y', (d) => d.y ? d.y - crossHeight / 2 : 0);
      // cross.append('rect')
      //     .attr('width', crossHeight)
      //     .attr('height', crossWidth)
      //     .attr('y', (d) => d.y ? d.y - crossWidth / 2 : 0)
      //     .attr('x', (d) => d.x ? d.x - crossHeight / 2 : 0);
    }

    console.log(`Rendered scatter layer in ${performance.now() - start}ms`);
    return () => {
      svg.selectAll('*').remove();
    };
  }, [scatterData, colormap, getHeight, localState.mapWidth, localState.layerAlpha.scatterMore, localState.layerAlpha.scatterOriginal, localState.scatterScale]);

  /**
 * Glyph layer
 */
  React.useEffect(() => {
    if (!glyphLayerRef.current) return;
    if (!projection) return;
    if (glyphDataGrid.length === 0 || glyphDataItem.length === 0) return;

    console.log('Rendering glyph layer');
    const start = performance.now();

    const svg = d3.select(glyphLayerRef.current);
    const width = localState.mapWidth;
    const height = getHeight();

    svg.attr('width', width).attr('height', height);

    const glyphScale = d3.scaleLinear([0, 3], [0.25, 2.5]);
    const glyphPathBack: string[] = [];
    const glyphPathFront: string[] = [];
    const glyphTransform: string[] = [];
    const glyphColorFront: string[] = [];

    // check whether the center of glyph is out of boundary. seemed useless
    const isOutside = (x: number, y: number) => {
      return isNaN(getSensorDensity(x / width, y / height));
    };

    const glyphSize = localState.glyph.mode === 'item' ? 10 : localState.gridSize;
    const Coord = localState.glyph.mode === 'item' ? filteredCoords(meta.coords, meta.known_set).map((d) => projection(d.coord as [number, number])) : gridCoords;
    const glyphData = localState.glyph.mode === 'item' ? glyphDataItem : glyphDataGrid;
    // console.log('glyphData', glyphData);
    for (let i = 0; i < glyphData.length; i++) {
      const center = Coord[i];
      if (glyphData[i] !== null && center && !isOutside(center[0], center[1])) {
        let strokeWidth = localState.gridSize / 40;
        const relativeScale = glyphData[i]!.glyphDirect * localState.glyph.relativity * glyphData[i]!.glyphUncert;
        // back start->top->end->bottom
        // consider turning glyphWidth into lpyghWWidth^2 to get better visual effect?
        // consider turning glyphHeight(hp) into log(p_x[1]+1) to get better visual effect?
        const renderWidth = glyphData[i]!.glyphWidth * glyphData[i]!.glyphWidth;
        const renderColor = glyphData[i]!.glyphUncert;
        const renderHeight = (h: number) => h > 0 ? Math.log10(1 + 9 * h) : -Math.log10(1 - 9 * h);
        // start
        const start = [0 - glyphSize / 2 * renderWidth, 0 + glyphSize / 2 * relativeScale];
        // end
        const p2 = [0 + glyphSize / 2 * renderWidth, 0 + glyphSize / 2 * relativeScale];
        // quartile
        let p3 = [0, 0 - glyphSize / 2 * renderHeight(glyphData[i]!.glythBottomQuartileVal)];
        let p1 = [0, 0 - glyphSize / 2 * renderHeight(glyphData[i]!.glythTopQuartileVal)];
        // confirm p3 is over p1
        if (p3[1] > p1[1]) {
          const temp = p3; p3 = p1; p1 = temp;
        }
        // inherent distance btw quartiles
        p3[1] -= strokeWidth * 2;
        p1[1] += strokeWidth * 2;
        // tanslate the left/right end

        glyphPathBack.push(`M ${start[0]} ${start[1]} L ${p1[0]} ${p1[1]} L ${p2[0]} ${p2[1]} L ${p3[0]} ${p3[1]} Z`);
        // front
        // avg
        const p4 = [0, 0 - glyphSize / 2 * renderHeight(glyphData[i]!.glythAvgVal)];
        // only One sensor in grid
        let m = strokeWidth / 2 * Math.sqrt(1 + Math.pow((p4[1] - start[1]) / (p4[0] - start[0]), 2));
        let x = start[0] * (p1[1] - p4[1] - m) / ((p1[1] - p4[1]));
        const k = (p4[1] - relativeScale * glyphSize / 2) / -start[0];
        // width gain
        if (Math.abs(k) > 1) {
          x = start[0] * 0.8;
          strokeWidth = d3.scaleLinear([0, 1], [1.4, 1.05])(Math.abs(1/k))*strokeWidth;
          m = strokeWidth / 2 * Math.sqrt(1 + Math.pow((p4[1] - start[1]) / (p4[0] - start[0]), 2));
        }
        const y1 = k * x + p4[1] + m;
        const y2 = k * x + p4[1] - m;
        const q1 = [x, y1];
        const q2 = [0, p4[1] + m];
        const q3 = [-q1[0], q1[1]];
        const q4 = [x, y2];
        const q5 = [0, p4[1] - m];
        const q6 = [-q4[0], q4[1]];
        glyphPathFront.push(`M ${start[0]} ${start[1]} L ${q1[0]} ${q1[1]} L ${q2[0]} ${q2[1]} L ${q3[0]} ${q3[1]} L ${p2[0]} ${p2[1]} L ${q6[0]} ${q6[1]} L ${q5[0]} ${q5[1]} L ${q4[0]} ${q4[1]} Z`);
        glyphTransform.push(`translate(${center![0]} ${center![1]}) scale(${glyphScale(localState.glyph.horizontalScale)} ${glyphScale(localState.glyph.verticalScale)}) scale(${glyphScale(localState.glyph.scale)})`);

        // color front
        // const colorGray = localState.glyph.interpUncertainty*(d3.scaleLinear([0, 1], [100, 0])(renderColor));
        // glyphColorFront.push(`rgba(${colorGray}, ${colorGray}, ${colorGray}, 1)`);
        const grayColormap = d3.scaleSequential(d3.interpolateGreys).domain([-0.5 - 3 * (1 - localState.glyph.interpUncertainty), 1]);
        const glyphColor = grayColormap(renderColor);
        glyphColorFront.push(glyphColor);
      } else {
        glyphPathBack.push('');
        glyphPathFront.push('');
        glyphTransform.push('');
        glyphColorFront.push('');
      }
    }
    const glyphConfig = glyphPathBack.map((d, i) => [d, glyphPathFront[i], glyphTransform[i], glyphColorFront[i]]);
    const glyphUnit = svg.selectAll('g')
        .data(glyphConfig)
        .enter().append('g')
        .attr('width', glyphSize)
        .attr('height', glyphSize)
        .attr('transform', (d) => d[2]);
    glyphUnit.append('path')
        .attr('d', (d) => d[0])
        .attr('stroke', 'none')
    // .attr('fill', 'rgba(220,220,220,0.5)');
        .attr('fill', 'rgba(200,200,200,0.5)');
    glyphUnit.append('path')
        .attr('d', (d) => d[1])
        .attr('d', (d) => d[1])
        .attr('stroke', 'none')
        .attr('fill', (d) => d[3]);

    console.log(`Rendered glyph layer in ${performance.now() - start}ms`);
    return () => {
      svg.selectAll('*').remove();
    };
  }, [projection, getSensorDensity, gridCoords, glyphDataGrid, glyphDataItem, localState.glyph, localState.gridSize, getHeight, localState.mapWidth, meta.coords, meta.known_set]);

  return (
    <div className={'text-sm'}>
      <div className={'flex flex-row w-full mb-2 justify-between items-center'}>
        <div className={'font-bold text-lg'}>{props.title}</div>
        <Button onClick={saveAsPDF}>Download</Button>
      </div>
      <div className={'border border-solid border-gray-300 relative p-4'} >
        <div style={{
          width: localState.mapWidth,
          height: getHeight(),
        }}>
          <div className='absolute' style={{
            width: localState.mapWidth,
            height: getHeight(),
            transform: `scale(${localState.layerAlpha.sx}, ${localState.layerAlpha.sy})`,
          }}>
            <Map
              id={`map-${props.title}`}
              mapStyle={'mapbox://styles/mapbox/navigation-day-v1'}
              // mapStyle={'mapbox://styles/mapbox/satellite-streets-v11'}
              attributionControl={false}
              mapboxAccessToken={'pk.eyJ1IjoiYmlsbGNoZW4yayIsImEiOiJjbGxnbW1lY3gweXdzM2hvM2o4em14a2R4In0.tyOAJUUzv68K6g2SjYVw3Q'}
              projection={{
                name: 'equirectangular',
              }}
              // @ts-ignore
              style={{
                display: 'absolute',
                width: localState.mapWidth,
                height: getHeight(),
                opacity: state.layerAlpha.tiles,
              }}
            ></Map>
          </div>
          {/* <svg ref={tileLayerRef} className={'absolute left-4 top-4'} opacity={state.layerAlpha.tiles}></svg> */}


          <canvas ref={heatLayerRef} className={'absolute'} style={{
            opacity: state.layerAlpha.heat,
            pointerEvents: 'none',

          }}></canvas>
          <canvas ref={hatchLayerRef} className={'absolute'} style={{
            opacity: state.layerAlpha.hatch,
            pointerEvents: 'none',
          }}></canvas>
          <svg ref={scatterLayerRef} className={'absolute'}
            pointerEvents={'none'}
            opacity={
              Math.min(state.layerAlpha.scatterOriginal, state.layerAlpha.scatterMore)
            }
          ></svg>
          <svg ref={terrainLayerRef}
            pointerEvents={'none'}
            className={'absolute'} opacity={state.layerAlpha.terrain}></svg>
          <svg ref={glyphLayerRef}
            pointerEvents={'none'}
            className={'absolute'} opacity={state.layerAlpha.glyph}></svg>
        </div>
        {/* <svg ref={tileLayerRef} className={'absolute'} opacity={state.layerAlpha.tiles}></svg> */}

        {/* {projection && */}

      </div>

      <div className={'flex flex-col gap-1'} style={{width: localState.mapWidth}}>
        {/* Information */}
        <div className={''}>
          {type === 'masked' &&
            <div>mse_masked: {mapMeta.mse_masked.toFixed(4)}, ssim_masked: {mapMeta.ssim_masked?.toFixed(4) || 'unknown'}</div>
          }
          {type === 'densified' &&
            <div>mse_densified: {mapMeta.mse_densified.toFixed(4)}, ssim_masked: {mapMeta.ssim_densified?.toFixed(4) || 'unknown'}</div>
          }
        </div>
        {/* Debug */}
        <div className={'text-blue-500'}>
          {[interpMapData, valPredictData, avgDistanceData, valTruthData, varianceData, sensorDensityData].map((one) => {
            return <div>
              {one.loading && one.currentParams && `Loading ${one.currentParams[0].split('/').splice(-1)}...`}
            </div>;
          })}
        </div>
        {/* Errors */}
        <div className={'text-red-500'}>
          {[interpMapData, valPredictData, avgDistanceData, valTruthData, varianceData, sensorDensityData].map((one) => {
            return <div>
              {one.error && one.currentParams && `Failed loading ${one.currentParams[0].split('/').splice(-1)}: ${one.error.message}`}
            </div>;
          })}
        </div>
      </div>
    </div>
  );
}
