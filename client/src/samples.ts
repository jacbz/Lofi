import sampleConfig from './samples.json';

export const SAMPLES_BASE_URL = './samples';
export const DRUM_LOOP_DEFAULT_VOLUME = -10;

class SampleGroup {
  name: string;

  category: string;

  urls: string[];

  bpm: number;

  volume: number;

  size: number;

  public constructor(name: string, category: string, urls: string[], bpm: number, volume: number) {
    this.name = name;
    this.category = category;
    this.urls = urls;
    this.bpm = bpm;
    this.volume = DRUM_LOOP_DEFAULT_VOLUME + volume;
    this.size = urls.length;
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
      sampleGroup.volume
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
