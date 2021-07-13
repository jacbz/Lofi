import * as Tonal from '@tonaljs/tonal';
import { Time } from 'tone/build/esm/core/type/Units';
import { Instrument } from './instruments';
import { OutputParams } from './params';

/**
 * A Track contains the elements that make up a lo-fi track.
 * Every Track has time signature 4/4.
 */
class Track {
  /** Root note of key, e.g. 'Db' */
  key: string;

  keyNum: number;

  /** Musical mode of key, e.g. 'major' or 'lydian' */
  mode: string;

  modeNum: number;

  /** Title of the track */
  title: string;

  /** Tempo in BPM (beats per minute) */
  bpm: number = 100;

  /** Whether to swing eighth notes */
  swing: boolean = false;

  /** Number of measures; each measure contains four beats */
  numMeasures: number = 60;

  /** Number of seconds to fade out at the end */
  fadeOutDuration: number = 10;

  /** Total length of the track in seconds */
  get length() {
    return Math.ceil(((this.numMeasures * 4) / this.bpm) * 60);
  }

  /** List of (sampleGroupName, sampleIndex) */
  samples: [string, number][];

  /** Sample loops */
  sampleLoops: SampleLoop[];

  /** Instruments to use, by name */
  instruments: Instrument[];

  /** Timings of notes */
  instrumentNotes: InstrumentNote[];

  /** Color of cover */
  color: string;

  /** The output params that generated this track */
  outputParams: OutputParams;

  public constructor(init?: Partial<Track>) {
    Object.assign(this, init);
  }
}

/**
 * Specifies a sample loop with a Tone.js start time and end time
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
  /** Instrument that should play the note */
  instrument: Instrument;

  /** Pitch(es) to play, e.g. 'D#1' or ['C', 'E', 'G'] */
  pitch: string | string[];

  /** Duration in Tone.js time, if null, play entire note */
  duration: Time;

  /** Onset time in Tone.js */
  time: Time;

  /** Velocity of the note, between 0 and 1 (defaults to 1) */
  velocity?: number;

  public constructor(
    instrument: Instrument,
    pitch: string | string[],
    duration: Time,
    time: Time,
    velocity?: number
  ) {
    this.instrument = instrument;
    this.pitch =
      typeof pitch === 'string'
        ? Tonal.Note.simplify(pitch)
        : pitch.map((p) => Tonal.Note.simplify(p));
    this.duration = duration;
    this.time = time;
    this.velocity = velocity;
  }
}

export { Track, SampleLoop, InstrumentNote };
