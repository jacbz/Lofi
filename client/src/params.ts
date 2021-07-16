export const LATENT_SPACE_DIMENSIONS = 100;
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
