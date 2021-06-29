import sampleConfig from './samples.json';

export const SAMPLES_BASE_URL = './samples';
export const DRUM_LOOP_DEFAULT_VOLUME = -10;

class SampleGroup {
  name: string;

  category: string;

  urls: string[];

  bpm: number[];

  volume: number;

  size: number;

  energyMap: number[][];

  public constructor(
    name: string,
    category: string,
    urls: string[],
    bpm: number[],
    volume: number,
    energyMap: number[][]
  ) {
    this.name = name;
    this.category = category;
    this.urls = urls;
    this.bpm = bpm;
    this.volume = DRUM_LOOP_DEFAULT_VOLUME + volume;
    this.size = urls.length;
    this.energyMap = energyMap;
  }

  getSampleUrl(index: number) {
    return `${SAMPLES_BASE_URL}/loops/${this.category}/${this.name}/${this.urls[index]}`;
  }
}

class SampleInstrument {
  name: String;

  volume: number;

  map: any;
}

export const LOOPS: Map<string, SampleGroup> = sampleConfig.loops.reduce((map, sampleGroup) => {
  map.set(
    sampleGroup.name,
    new SampleGroup(
      sampleGroup.name,
      sampleGroup.category,
      sampleGroup.urls,
      sampleGroup.bpm,
      sampleGroup.volume,
      sampleGroup.energyMap
    )
  );
  return map;
}, new Map());

export const SAMPLE_INSTRUMENTS: Map<string, SampleInstrument> = sampleConfig.instruments.reduce(
  (map, instrument) => {
    map.set(instrument.name, instrument);
    return map;
  },
  new Map()
);

export const selectDrumbeat = (bpm: number, energy: number): [string, number] => {
  let bpmGroup = Math.round(bpm / 5) * 5;
  if (bpmGroup < 70) bpmGroup = 70;
  if (bpmGroup > 100) bpmGroup = 100;
  const sampleGroup = `drumloop${bpmGroup}`;

  const index = LOOPS.get(sampleGroup).energyMap.findIndex(
    (range) => range[0] <= energy && range[1] >= energy
  );

  return [sampleGroup, index];
};
