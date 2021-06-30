import sampleConfig from './samples.json';

export const SAMPLES_BASE_URL = './samples';
export const DRUM_LOOP_DEFAULT_VOLUME = -6;

class SampleGroup {
  name: string;

  category: string;

  urls: string[];

  volume: number;

  size: number;

  energyRanges: number[][];

  public constructor(
    name: string,
    category: string,
    urls: string[],
    volume: number,
    energyRanges: number[][]
  ) {
    this.name = name;
    this.category = category;
    this.urls = urls;
    this.volume = DRUM_LOOP_DEFAULT_VOLUME + volume;
    this.size = urls.length;
    this.energyRanges = energyRanges;
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
      sampleGroup.volume,
      sampleGroup.energyRanges
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
  const sampleGroup = `drumloop${bpm}`;

  const index = LOOPS.get(sampleGroup).energyRanges.findIndex(
    (range) => range[0] <= energy && range[1] >= energy
  );

  return [sampleGroup, index];
};
