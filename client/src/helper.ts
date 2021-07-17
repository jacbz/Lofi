/**
 * Helper classes and functions
 */
import * as Tone from 'tone';
import { Time } from 'tone/build/esm/core/type/Units';
import * as Tonal from '@tonaljs/tonal';
import { deflate, inflate } from 'pako';

/** Wraps inaccessible Tonal.Chord class */
export class Chord {
  empty: boolean;

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
  if (!octaves) return note;
  const n = Tonal.Note.get(note);
  return `${n.pc}${n.oct + octaves}`;
};

/** Shifts given notes by a number of octaves */
export const octShiftAll = (notes: string[], octaves: number) =>
  notes.map((n) => octShift(n, octaves));

/** Maps a given note number to a scale degree and octave, e.g. 8 -> [0, 1] */
export const mapNote = (note: number) => {
  const scaleDegreeIndex = (note - 1) % 7;
  const octave = Math.floor((note - 1) / 7);
  return [scaleDegreeIndex, octave];
};

/** Mounts given note numbers on a given scale */
export const mountNotesOnScale = (offsetScaleDegree: number, notes: number[], scale: string[]) =>
  notes.map((n) => {
    const [scaleDegreeIndex, octave] = mapNote(n + offsetScaleDegree - 1);
    return octShift(scale[scaleDegreeIndex], octave);
  });

/** Converts a key number to string, e.g. 2 => 'C#' */
export const keyNumberToString = (key: number): string =>
  Tonal.Scale.get('C chromatic').notes[key - 1];

/** Adds two Tone.js Time objects together */
export const addTime = (time1: Time, time2: Time) => {
  const time = Tone.Time(time1).toSeconds() + Tone.Time(time2).toSeconds();
  return Tone.Time(time).toBarsBeatsSixteenths();
};

/** Substracts one Tone.js Time objects to another */
export const subtractTime = (time1: Time, time2: Time) => {
  const time = Tone.Time(time1).toSeconds() - Tone.Time(time2).toSeconds();
  return Tone.Time(time).toBarsBeatsSixteenths();
};

/** Converts a number of measures to seconds */
export const measuresToSeconds = (measures: number, bpm: number) => {
  const measureInSeconds = 240 / bpm;
  return measureInSeconds * measures;
};

/** Returns a number sampled from a standard normal distribution using the Box–Muller transform */
export const randn = () => {
  let u = 0;
  let v = 0;
  while (u === 0) u = Math.random(); // Converting [0,1) to (0,1)
  while (v === 0) v = Math.random();
  return Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
};

/** Returns a quasi-random number between min-max based on given seed number */
export const randomFromInterval = (min: number, max: number, seed?: number) => {
  const randomNumber = seed ? random(seed) : Math.random();
  return Math.floor(randomNumber * (max - min + 1) + min);
};

/** Returns a quasi-random number between 0-1 based on given seed number */
export const random = (seed: number) => {
  const x = Math.sin(seed) * 10000;
  return x - Math.floor(x);
};

/** Generates a random pastel color based on seed */
export const randomColor = (seed: number) =>
  `hsl(${360 * random(seed)},${25 + 70 * random(seed + 1)}%,${85 + 10 * random(seed + 2)}%)`;

/** Compresses a given string into a Base64-encoded string using deflate */
export const compress = (input: string) => btoa(String.fromCharCode.apply(null, deflate(input)));

/** Decompresses a given Base64-encoded string using inflate */
export const decompress = (input: string) =>
  inflate(
    Uint8Array.from(atob(input), (c) => c.charCodeAt(0)),
    { to: 'string' }
  );
