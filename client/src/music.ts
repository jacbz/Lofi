/**
 * Helper classes for music
 */
import * as Tonal from '@tonaljs/tonal';

/** Wraps inaccessible Tonal.Chord class */
export class Chord {
  name: string;

  aliases: string[];

  tonic: string | null;

  type: string;

  root: string;

  rootDegree: number;

  symbol: string;

  notes: Tonal.NoteName[];
}

/** Shifts a given note by a number of octaves */
export const octShift = (note: string, octaves: number) => {
  const n = Tonal.Note.get(note);
  return Tonal.Note.get(`${n.pc}${n.oct + octaves}`).name;
};

export default Chord;
