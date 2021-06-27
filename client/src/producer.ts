import { Chord, Mode, Note, Scale } from '@tonaljs/tonal';
import { Track, Loop, Timing } from './track';
import { OutputParams } from './params';

abstract class Producer {
  static toTime(measure: number, beat: number) {
    return `${measure}:${beat}`;
  }

  static produce(params: OutputParams): Track {
    // tonic note, e.g. 'G'
    const tonic = Scale.get('C chromatic').notes[params.key - 1];

    // musical mode, e.g. 'ionian'
    const mode = Mode.names()[params.mode - 1];

    // array of notes, e.g. ["C", "D", "E", "F", "G", "A", "B"]
    const notes = Mode.notes(mode, tonic);

    // array of triads, e.g. ["C", "Dm", "Em", "F", "G", "Am", "Bdim"]
    const chords = Mode.seventhChords(mode, tonic);
    console.log(chords);

    const numberOfIterations = 6;

    const numMeasures = params.chordProgression.length * numberOfIterations;

    const instruments = ['guitar-bass', 'piano', 'guitar-electric'];
    const noteTimings: Timing[] = [];
    for (let i = 0; i < numberOfIterations; i += 1) {
      for (let chordNo = 0; chordNo < params.chordProgression.length; chordNo += 1) {
        const measure = i * params.chordProgression.length + chordNo;
        const chordIndex = params.chordProgression[chordNo] - 1;
        const chordString = chords[chordIndex];
        // e.g. Chord.getChord("maj7", "G4")
        const chord = Chord.getChord(Chord.get(chordString).aliases[0], `${notes[chordIndex]}3`);

        // bass line
        const rootNote = Mode.notes(mode, `${tonic}1`)[chordIndex];
        const bassTiming = new Timing('guitar-bass', rootNote, '1m', this.toTime(measure, 0));
        noteTimings.push(bassTiming);

        for (let note = 0; note < 4; note += 1) {
          noteTimings.push(
            new Timing(
              i % 2 === 0 ? 'piano' : 'guitar-electric',
              Note.simplify(chord.notes[note]),
              '0:3',
              this.toTime(measure, note * 0.25 + 1)
            )
          );
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
