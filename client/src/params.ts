export const HIDDEN_SIZE = 100;
export const DEFAULT_OUTPUTPARAMS = 'eJyFVltu3DAMvIv7KxTiS6JylcV+tInRBk2aIg0KFEXuXtqWvHqsN6AdZymJFkfDoU//prfHt6d5ups+QdQUiCSEhICMGhIHZp7c9GP+O92Rm55fHmwquOnrr+fpTu2f+ef8+s0G/efI6KY/X57mn/fz8jtgdNP995fXh9/T3QlccOLAsd1ytkjz08vD47wMnZLT2s7uFFxj2bOMgl/+gD3NqQcLl/HLZe51VTtTbCubxcXO53dXYwFRVNR7oESQEpCQiGjBAgoWIWMRscYCVRoskKHBQlYc9meHh5q/snWzvUe7JNtU63zb9I8jXot3iF792zvfoYeRvffs1VAEDBACR6G0Myn2TIpSo2cUbNDzqSMSbTSy50Konk5jaiOdyBXz6/4vcy450rqehmPwxao4Fw84PXi7X2+/xvG19ehpiFGFQQgoKAuyxxiloCdDHaYavcAteqTawLcAuLEvDNBF2/xSp2FlJucESzLsaE2n3bw7eRvxH8Dih1XX4tyCBTz4aNKkmGLyPggLYyLcSxILLlJYFWtcQLSrSexoxRkXyOhsHh6qcxSdWFmtX1Y55dqLpvL1xVkCxiuexWcp5stW47IWW9/lXc0I1iOh20Pt74t8mJeDX3HfHmmP0uQAWEEjSdKomgCFWXeGDyepvmG4h+YkpWF4zLoaF2kYtHXjFhTxXwsWDniYWW8evsHVEifszLnMwdryu+CQ4YTGZybruzGRdWFTgBCpYvjYdRpcEHqGt5VPxkijgt2bAuBVbKSh3dh39i6SPcuavrVe619HDeS4I33s6YTTlJIYTR1CChJUSGP0SqNC7H2HGvxi13c8VfjhyivJfdvfaDoFBarNPDjwYdSSZk1GqsQ9jtPXbqeczEgQWBkUraFAghRQ0g1cGuXshZNq4QyZTGGTzA4Wq/5Un/QmEddJlb8lcoccPk98+arx1awevlp2qp6M2ydj9Yrz+/k/KIyMcg==';
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
