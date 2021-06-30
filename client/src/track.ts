import * as Tonal from '@tonaljs/tonal';
import { Time } from 'tone/build/esm/core/type/Units';
import { Instrument } from './instruments';

/**
 * A Track contains the elements that make up a lo-fi track.
 * Every Track has time signature 4/4.
 */
class Track {
  /** Root note of key, e.g. 'Db' */
  key: string;

  /** Musical mode of key, e.g. 'major' or 'lydian' */
  mode: string;

  /** Title of the track */
  title: string;

  /** How energetic the track should be, 0 (less energetic) to 1 (very energetic) */
  energy: number;

  /** How positive the music should be, 0 (sad) to 1 (cheerful) */
  valence: number;

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
  instruments: Instrument[];

  /** Timings of notes */
  instrumentNotes: InstrumentNote[];

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

  /** Duration in Tone.js time */
  duration: Time;

  /** Onset time in Tone.js */
  time: Time;

  public constructor(instrument: Instrument, pitch: string | string[], duration: Time, time: Time) {
    this.instrument = instrument;
    this.pitch =
      typeof pitch === 'string'
        ? Tonal.Note.simplify(pitch)
        : pitch.map((p) => Tonal.Note.simplify(p));
    this.duration = duration;
    this.time = time;
  }
}

export { Track, SampleLoop, InstrumentNote };
