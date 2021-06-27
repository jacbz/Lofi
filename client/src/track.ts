import { Mode, Chord } from '@tonaljs/tonal/dist/index';
import * as Tone from 'tone';
import { Time } from 'tone/build/esm/core/type/Units';
/**
 * A Track contains the elements that make up a lo-fi track.
 * Every Track has time signature 4/4.
 */
class Track {
  key: string;

  mode: string;

  /** Tempo in BPM (beats per minute) */
  bpm: number = 100;

  /** Number of measures; each measure contains four beats */
  numMeasures: number = 60;

  /** Total length of the track in seconds */
  get length() {
    return Math.ceil(((this.numMeasures * 4) / this.bpm) * 60);
  }

  /** Loops to use, by sample id */
  loopIds: number[];

  /** Timings of the sample loops */
  loops: Loop[];

  /** Instruments to use, by name */
  instruments: string[];

  /** Timings of notes */
  noteTimings: Timing[];

  public constructor(init?: Partial<Track>) {
    Object.assign(this, init);
  }
}

/**
 * Specifies a loop with a Tone.js start time and end time
 */
class Loop {
  /** Id of the sample */
  sampleId: number;

  /** Onset time in Tone.js */
  startTime: Time;

  /** Stop time in Tone.js */
  stopTime: Time;

  public constructor(sampleId: number, startTime: Time, stopTime: Time) {
    this.sampleId = sampleId;
    this.startTime = startTime;
    this.stopTime = stopTime;
  }
}

/**
 * Precise timing of a single note played by an instrument
 */
class Timing {
  /** Name of the instrument */
  instrument: string;

  /** Pitch(es) of the instrument, e.g. D#1 or [C, E, G] */
  pitch: string | string[];

  /** Duration in Tone.js time */
  duration: Time;

  /** Onset time in Tone.js */
  time: Time;

  public constructor(instrument: string, pitch: string | string[], duration: Time, time: Time) {
    this.instrument = instrument;
    this.pitch = pitch;
    this.duration = duration;
    this.time = time;
  }
}

export { Track, Loop, Timing };
