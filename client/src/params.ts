import { randomFromInterval } from './helper';

export const getRandomOutputParams = () => {
  const params: OutputParams = {
    title: null,
    // key: 8,
    key: randomFromInterval(1, 12),
    mode: 6,
    // mode: Math.random() < 0.5 ? 6 : 1,
    bpm: randomFromInterval(70, 90),
    energy: Math.random(),
    valence: Math.random(),
    chords: [1, 4, 6, 5, 1, 4, 6, 5],
    melodies: [] as number[][]
  };
  return JSON.stringify(params, null, 2);
};

export const HIDDEN_SIZE = 100;
export class OutputParams {
  /** Optional: title */
  title: string;

  /** Key as a number between 1-12 */
  key: number;

  /**
   * Musical mode
   * 1: Ionian (Major)
   * 2: Dorian
   * 3: Phrygia
   * 4: Lydian
   * 5: Mixolydian
   * 6: Aeolian (Minor)
   * 7: Locrian
   */
  mode: number;

  /** Beats per minute */
  bpm: number;

  /** How energetic the track should be, 0 (less energetic) to 1 (very energetic) */
  energy: number;

  /** How positive the music should be, 0 (sad) to 1 (cheerful) */
  valence: number;

  chords: number[];

  melodies: number[][];
}
