import { Chord, Mode, Note, Scale } from '@tonaljs/tonal';
import { Track, Loop, Timing } from './track';
import { OutputParams } from './params';

abstract class Producer {
  static produceExampleTrack(): Track {
    const track = new Track({
      key: null as any,
      mode: null as any,
      bpm: 70,
      numMeasures: 6,
      loopIds: [1],
      loops: [new Loop(1, '1:0', '5:0')]
    });
    return track;
  }

  static toTime(measure: number, beat: number) {
    return `${measure}:${beat}`;
  }

  static produce(params: OutputParams): Track {
    // tonic note, e.g. 'G'
    const tonic = Note.transpose('C', `${params.key}m`);

    // musical mode, e.g. 'ionian'
    const mode = Mode.names()[params.mode - 1];

    // array of notes, e.g. ["C", "D", "E", "F", "G", "A", "B"]
    const notes = Mode.notes(mode, tonic);

    // array of triads, e.g. ["C", "Dm", "Em", "F", "G", "Am", "Bdim"]
    const triads = Mode.triads(mode, tonic);

    const numberOfIterations = 6;

    const numMeasures = params.chordProgression.length * numberOfIterations;

    const instruments = ['Bass1'];
    const noteTimings: Timing[] = [];
    for (let i = 0; i < numberOfIterations; i += 1) {
      for (let chordNo = 0; chordNo < params.chordProgression.length; chordNo += 1) {
        const measure = i * 4 + chordNo;
        const rootNoteIndex = params.chordProgression[chordNo].sd - 1;
        const bassTiming = new Timing('Bass1', `${notes[rootNoteIndex]}1`, '1m', this.toTime(measure, 0));
        noteTimings.push(bassTiming);
      }
    }

    const track = new Track({
      mode,
      key: tonic,
      numMeasures,
      bpm: 80,
      loopIds: [1],
      loops: [new Loop(1, '0:0', `${numMeasures}:0`)],
      instruments,
      noteTimings
    });
    return track;
  }
}

export default Producer;
