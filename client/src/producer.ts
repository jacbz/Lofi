import { Chord, Mode, Note, Scale } from '@tonaljs/tonal';
import { InstrumentNote, SampleLoop, Track } from './track';
import { OutputParams } from './params';
import { LOOPS } from './samples';

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
    const instrumentNotes: InstrumentNote[] = [];
    for (let i = 0; i < numberOfIterations; i += 1) {
      for (let chordNo = 0; chordNo < params.chordProgression.length; chordNo += 1) {
        const measure = i * params.chordProgression.length + chordNo;
        const chordIndex = params.chordProgression[chordNo] - 1;
        const chordString = chords[chordIndex];
        // e.g. Chord.getChord("maj7", "G4")
        const chord = Chord.getChord(Chord.get(chordString).aliases[0], `${notes[chordIndex]}3`);

        // bass line
        const rootNote = Mode.notes(mode, `${tonic}1`)[chordIndex];
        const bassTiming = new InstrumentNote('guitar-bass', rootNote, '1m', this.toTime(measure, 0));
        instrumentNotes.push(bassTiming);

        for (let note = 0; note < 4; note += 1) {
          instrumentNotes.push(
            new InstrumentNote(
              i % 2 === 0 ? 'piano' : 'guitar-electric',
              Note.simplify(chord.notes[note]),
              '0:3',
              this.toTime(measure, note * 0.25 + 1)
            )
          );
        }
      }
    }

    const { energy, valence } = params;

    const samples: [string, number][] = [];
    const sampleLoops: SampleLoop[] = [];

    samples.push(['drumloop1', 0]);
    sampleLoops.push(new SampleLoop('drumloop1', 0, '0:0', `${numMeasures}:0`));

    const randomVinyl = this.randomFromInterval(0, LOOPS.get('vinyl').size - 1, energy + valence);
    samples.push(['vinyl', randomVinyl]);
    sampleLoops.push(new SampleLoop('vinyl', randomVinyl, '0:0', `${numMeasures}:0`));

    const track = new Track({
      mode,
      key: tonic,
      energy,
      valence,
      numMeasures,
      bpm: 80,
      samples,
      sampleLoops,
      instruments,
      instrumentNotes
    });
    return track;
  }

  static randomFromInterval(min: number, max: number, seed: number) {
    return Math.floor(this.random(seed) * (max - min + 1) + min);
  }

  /** Returns a quasi-random number between 0-1 based on given seed number */
  static random(seed: number) {
    const x = Math.sin(seed) * 10000;
    return x - Math.floor(x);
  }
}

export default Producer;
