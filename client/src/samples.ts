import * as Tone from 'tone';
import { keyNumberToString, pitchShiftDistance, random, randomFromInterval } from './helper';
import sampleConfig from './samples.json';

export const SAMPLES_BASE_URL = './samples';
export const SAMPLE_DEFAULT_VOLUME = -8;

class SampleGroup {
  name: string;

  isLoop: boolean;

  volume: number;

  size: number;

  energyRanges?: number[][];

  mode: number;

  keys?: number[];

  durations?: number[];

  public constructor(
    name: string,
    isLoop: boolean,
    size: number,
    volume: number,
    energyRanges?: number[][],
    mode?: number,
    keys?: number[],
    durations?: number[]
  ) {
    this.name = name;
    this.isLoop = isLoop;
    this.volume = SAMPLE_DEFAULT_VOLUME + volume;
    this.energyRanges = energyRanges;
    this.size = size;
    this.mode = mode;
    this.keys = keys;
    this.durations = durations;
  }

  /** Gets a random sample index, based on a seed number */
  getRandomSample(seed: number) {
    return randomFromInterval(0, this.size - 1, seed);
  }

  getRandomSampleByKey(seed: number, key: number) {
    const weights = this.keys.map((k) =>
      1 / (Math.abs(pitchShiftDistance(keyNumberToString(k), keyNumberToString(key))) + 1) ** 4);
    return this.getRandomSampleWithWeights(seed, weights);
  }

  /** Gets a random sample index, weighted.
   * Weights must be of length size */
  getRandomSampleWithWeights(seed: number, weights: number[]) {
    const weightsSum = weights.reduce((pv, cv) => pv + cv, 0);
    const lastIndex = this.size - 1;

    const num = random(seed);

    let s = 0;
    for (let i = 0; i < lastIndex; i += 1) {
      s += weights[i] / weightsSum;
      if (num < s) {
        return i;
      }
    }
    return lastIndex;
  }

  getSampleUrl(index: number) {
    return `${SAMPLES_BASE_URL}/${this.isLoop ? 'loops' : 'oneshots'}/${this.name}/${this.name}_${
      index + 1
    }.mp3`;
  }

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

export const SAMPLEGROUPS: Map<string, SampleGroup> = new Map([
  ...sampleConfig.loops.reduce((map, sampleGroup) => {
    map.set(
      sampleGroup.name,
      new SampleGroup(
        sampleGroup.name,
        true,
        sampleGroup.size,
        sampleGroup.volume,
        sampleGroup.energyRanges
      )
    );
    return map;
  }, new Map()),
  ...sampleConfig.oneshots.reduce((map, sampleGroup) => {
    map.set(
      sampleGroup.name,
      new SampleGroup(
        sampleGroup.name,
        false,
        sampleGroup.keys.length,
        sampleGroup.volume,
        null,
        sampleGroup.mode,
        sampleGroup.keys,
        sampleGroup.durations
      )
    );
    return map;
  }, new Map())
]);

export const selectDrumbeat = (bpm: number, energy: number): [string, number] => {
  const sampleGroup = `drumloop${bpm}`;

  const index = SAMPLEGROUPS.get(sampleGroup).energyRanges.findIndex(
    (range) => range[0] <= energy && range[1] >= energy
  );

  return [sampleGroup, index];
};
