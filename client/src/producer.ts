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
    const chords = Mode.seventhChords(mode, tonic);

    const numberOfIterations = 6;

    const numMeasures = params.chordProgression.length * numberOfIterations;

    const instruments = ['bass1', 'piano'];
    const noteTimings: Timing[] = [];
    for (let i = 0; i < numberOfIterations; i += 1) {
      for (let chordNo = 0; chordNo < params.chordProgression.length; chordNo += 1) {
        const measure = i * 4 + chordNo;
        const chordIndex = params.chordProgression[chordNo].sd - 1;

        // bass line
        const bassTiming = new Timing('bass1', `${notes[chordIndex]}1`, '1m', this.toTime(measure, 0));
        noteTimings.push(bassTiming);

        const chordString = chords[chordIndex];
        const chord = Chord.getChord(chordString.substring(1), `${notes[chordIndex]}3`);
        for (let note = 0; note < 4; note += 1) {
          noteTimings.push(new Timing('piano', chord.notes[note], '0:3', this.toTime(measure, note * 0.25 + 1)));
        }
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
