/**
 * Helper classes and functions
 */
import * as Tone from 'tone';
import { Time } from 'tone/build/esm/core/type/Units';
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
  return `${n.pc}${n.oct + octaves}`;
};

/** Adds two Tone.js Time objects together */
export const addTime = (time1: Time, time2: Time) => {
  const time = Tone.Time(time1).toSeconds() + Tone.Time(time2).toSeconds();
  return Tone.Time(time).toBarsBeatsSixteenths();
};

export const subtractTime = (time1: Time, time2: Time) => {
  const time = Tone.Time(time1).toSeconds() - Tone.Time(time2).toSeconds();
  return Tone.Time(time).toBarsBeatsSixteenths();
};

/** Returns a quasi-random number between min-max based on given seed number */
export const randomFromInterval = (min: number, max: number, seed: number) => {
  const randomNumber = random(seed);
  return Math.floor(randomNumber * (max - min + 1) + min);
};

/** Returns a quasi-random number between 0-1 based on given seed number */
export const random = (seed: number) => {
  const x = Math.sin(seed) * 10000;
  return x - Math.floor(x);
};
