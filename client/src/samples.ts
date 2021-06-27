import sampleConfig from './samples.json';

export const SAMPLES_BASE_URL = './samples/';
export const DRUM_LOOP_DEFAULT_VOLUME = -10;

class Sample {
  name: string;

  url: string;

  bpm: number;

  volume: number;

  public constructor(name: string, url: string, bpm: number, volume: number) {
    this.name = name;
    this.url = SAMPLES_BASE_URL + url;
    this.bpm = bpm;
    this.volume = DRUM_LOOP_DEFAULT_VOLUME + volume;
  }
}

class SampleInstrument {
  name: String;

  volume: number;

  map: any;
}

/** Drum loop configs as loaded from samples.json */
export const DRUM_LOOPS: Map<number, Sample> = sampleConfig.drum_loops.reduce((map, drumLoop) => {
  map.set(drumLoop.id, new Sample(drumLoop.name, drumLoop.url, drumLoop.bpm, drumLoop.volume));
  return map;
}, new Map());

export const SAMPLE_INSTRUMENTS: Map<string, SampleInstrument> = sampleConfig.instruments.reduce(
  (map, instrument) => {
    map.set(instrument.name, instrument);
    return map;
  },
  new Map()
);
