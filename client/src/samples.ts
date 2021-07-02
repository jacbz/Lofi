import * as Tone from 'tone';
import { randomFromInterval } from './helper';
import sampleConfig from './samples.json';

export const SAMPLES_BASE_URL = './samples';
export const SAMPLE_DEFAULT_VOLUME = -8;

class SampleGroup {
  name: string;

  category: string;

  volume: number;

  energyRanges: number[][];

  size: number;

  public constructor(
    name: string,
    category: string,
    size: number,
    energyRanges: number[][],
    volume: number
  ) {
    this.name = name;
    this.category = category;
    this.volume = SAMPLE_DEFAULT_VOLUME + volume;
    this.energyRanges = energyRanges;
    this.size = size;
  }

  /** Gets a random sample index, based on a seed number */
  getRandomSample(seed: number) {
    return randomFromInterval(0, this.size - 1, seed);
  }

  getSampleUrl(index: number) {
    return `${SAMPLES_BASE_URL}/loops/${this.category}/${this.name}/${this.name}_${index + 1}.mp3`;
  }

  getFilters() {
    if (this.category === 'drums') {
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

export const SAMPLEGROUPS: Map<string, SampleGroup> = sampleConfig.reduce((map, sampleGroup) => {
  map.set(
    sampleGroup.name,
    new SampleGroup(
      sampleGroup.name,
      sampleGroup.category,
      sampleGroup.size,
      sampleGroup.energyRanges,
      sampleGroup.volume
    )
  );
  return map;
}, new Map());

export const selectDrumbeat = (bpm: number, energy: number): [string, number] => {
  const sampleGroup = `drumloop${bpm}`;

  const index = SAMPLEGROUPS.get(sampleGroup).energyRanges.findIndex(
    (range) => range[0] <= energy && range[1] >= energy
  );

  return [sampleGroup, index];
};
