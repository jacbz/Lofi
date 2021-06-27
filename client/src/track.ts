import { Mode, Chord } from '@tonaljs/tonal/dist/index';
import * as Tone from 'tone';
import { Time } from 'tone/build/esm/core/type/Units';
/**
 * A Track contains the elements that make up a lo-fi track.
 * Every Track has time signature 4/4.
 */
class Track {
  /** Root note of key, e.g. 'Db' */
  key: string;

  /** Musical mode of key, e.g. 'major' or 'lydian' */
  mode: string;

  /** How energetic the track should be, 0 (less energetic) to 1 (very energetic) */
  energy: number

  /** How positive the music should be, 0 (sad) to 1 (cheerful) */
  valence: number

  /** Tempo in BPM (beats per minute) */
  bpm: number = 100;

  /** Number of measures; each measure contains four beats */
  numMeasures: number = 60;

  /** Total length of the track in seconds */
  get length() {
    return Math.ceil(((this.numMeasures * 4) / this.bpm) * 60);
  }

  /** List of (sampleGroupName, sampleIndex) */
  samples: [string, number][];

  /** Sample loops */
  sampleLoops: SampleLoop[];

  /** Instruments to use, by name */
  instruments: string[];

  /** Timings of notes */
  instrumentNotes: InstrumentNote[];

  public constructor(init?: Partial<Track>) {
    Object.assign(this, init);
  }
}

/**
 * Specifies a loop with a Tone.js start time and end time
 */
class SampleLoop {
  /** Name of the sample group */
  sampleGroupName: string;

  /** Index within sample group */
  sampleIndex: number;

  /** Onset time in Tone.js */
  startTime: Time;

  /** Stop time in Tone.js */
  stopTime: Time;

  public constructor(sample: string, sampleIndex: number, startTime: Time, stopTime: Time) {
    this.sampleGroupName = sample;
    this.sampleIndex = sampleIndex;
    this.startTime = startTime;
    this.stopTime = stopTime;
  }
}

/**
 * Precise timing of a single note played by an instrument
 */
class InstrumentNote {
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

export { Track, SampleLoop, InstrumentNote };
