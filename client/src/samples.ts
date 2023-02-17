import * as Tone from 'tone';
import { randomFromInterval } from './helper';
import sampleConfig from './samples.json';

export const SAMPLES_BASE_URL = './samples';
export const SAMPLE_DEFAULT_VOLUME = -6;

/** A SampleGroup defines a collection of samples, as taken from samples.json */
class SampleGroup {
  name: string;

  volume: number;

  size: number;

  energyRanges?: number[][];

  public constructor(name: string, size: number, volume: number, energyRanges?: number[][]) {
    this.name = name;
    this.volume = SAMPLE_DEFAULT_VOLUME + volume;
    this.energyRanges = energyRanges;
    this.size = size;
  }

  /** Gets a random sample index, based on a seed number */
  getRandomSample(seed: number) {
    return randomFromInterval(0, this.size - 1, seed);
  }

  getSampleUrl(index: number) {
    // for drumloop100 we have a single file
    if(this.name === 'drumloop100') {
      return `${SAMPLES_BASE_URL}/loops/${this.name}/${this.name}.mp3`;  
    }
    return `${SAMPLES_BASE_URL}/loops/${this.name}/${this.name}_${index + 1}.mp3`;
  }

  /** Returns sample-specific Tone.js filters */
  getFilters(): any[] {
    if (this.name.includes('drumloop')) {
      return [
        new Tone.Filter({
          type: 'lowpass',
          frequency: 2400,
          Q: 0.5
        })
      ];
    }
    return [];
  }
}

export const SAMPLEGROUPS: Map<string, SampleGroup> = sampleConfig.loops.reduce(
  (map, sampleGroup) => {
    map.set(
      sampleGroup.name,
      new SampleGroup(
        sampleGroup.name,
        sampleGroup.size,
        sampleGroup.volume,
        sampleGroup.energyRanges
      )
    );
    return map;
  },
  new Map()
);

/** Selects a suitable drumbeat based on BPM and energy value */
export const selectDrumbeat = (bpm: number, energy: number): [string, number] => {
  const sampleGroup = `drumloop${bpm}`;

  const index = SAMPLEGROUPS.get(sampleGroup).energyRanges.findIndex(
    (range) => range[0] <= energy && range[1] >= energy
  );

  return [sampleGroup, index];
};
