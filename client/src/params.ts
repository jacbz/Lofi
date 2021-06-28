export const DEFAULT_OUTPUT_PARAMS = {
  key: 8,
  // key: Math.floor(Math.random() * 12 + 1),
  mode: 6,
  // mode: Math.random() < 0.5 ? 6 : 1,
  energy: Math.random(),
  valence: Math.random(),
  chordProgression: [1, 4, 6, 5]
};

export class InputParams {
  text: string;
}

export class OutputParams {
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

  /** How energetic the track should be, 0 (less energetic) to 1 (very energetic) */
  energy: number;

  /** How positive the music should be, 0 (sad) to 1 (cheerful) */
  valence: number;

  chordProgression: number[];
}
