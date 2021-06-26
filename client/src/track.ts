import * as Tone from 'tone';
import { Time } from 'tone/build/esm/core/type/Units';

/**
 * A Track contains the elements that make up a lo-fi track.
 * Every Track has time signature 4/4.
 */
class Track {
  key: any;

  mode: any;

  /** Tempo in BPM (beats per minute) */
  bpm: number = 100;

  /** Number of measures; each measure contains four beats */
  numMeasures: number = 60;

  /** Total length of the track in seconds */
  get length() {
    return ((this.numMeasures * 4) / this.bpm) * 60;
  }

  /** Drum loops of the track, as a tuple list of (drum loop id, loop parameters) */
  drumLoops: [number, Loop][];

  public constructor(init?: Partial<Track>) {
    Object.assign(this, init);
  }
}

/**
 * Specifies a loop with a Tone.js start time and end time
 */
class Loop {
  /** Onset of the loop in measures */
  startTime: Time;

  /** Number of measures */
  stopTime: Time;

  public constructor(startTime: Time, stopTime: Time) {
    this.startTime = startTime;
    this.stopTime = stopTime;
  }
}

export { Track, Loop };
