import { Chord, Mode, Note, Scale } from '@tonaljs/tonal';
import { InstrumentNote, SampleLoop, Track } from './track';
import { OutputParams } from './params';
import { LOOPS } from './samples';

class Producer {
  tonic: string;

  mode: string;

  energy: number;

  valence: number;

  notesInScale: string[];

  chordsInScale: string[];

  chordProgression: number[];

  numberOfIterations = 6;

  numMeasures: number;

  samples: [string, number][] = [];

  sampleLoops: SampleLoop[] = [];

  instruments: string[] = [];

  instrumentNotes: InstrumentNote[] = [];

  static toTime(measure: number, beat: number) {
    return `${measure}:${beat}`;
  }

  produce(params: OutputParams): Track {
    // tonic note, e.g. 'G'
    this.tonic = Scale.get('C chromatic').notes[params.key - 1];

    // musical mode, e.g. 'ionian'
    this.mode = Mode.names()[params.mode - 1];

    // array of notes, e.g. ["C", "D", "E", "F", "G", "A", "B"]
    this.notesInScale = Mode.notes(this.mode, this.tonic);

    // array of triads, e.g. ["C", "Dm", "Em", "F", "G", "Am", "Bdim"]
    this.chordsInScale = Mode.seventhChords(this.mode, this.tonic);
    console.log(this.chordsInScale);

    this.numberOfIterations = 6;
    this.numMeasures = params.chordProgression.length * this.numberOfIterations;
    this.energy = params.energy;
    this.valence = params.valence;
    this.chordProgression = params.chordProgression;

    this.produceFx();
    this.produceIntro();
    this.produceMain();
    this.produceOutro();

    const track = new Track({
      mode: this.mode,
      key: this.tonic,
      energy: this.energy,
      valence: this.valence,
      numMeasures: this.numMeasures,
      bpm: 80,
      samples: this.samples,
      sampleLoops: this.sampleLoops,
      instruments: this.instruments,
      instrumentNotes: this.instrumentNotes
    });
    return track;
  }

  produceFx() {
    // vinyl crackle
    const randomVinyl = Producer.randomFromInterval(0, LOOPS.get('vinyl').size - 1, this.energy + this.valence);
    this.samples.push(['vinyl', randomVinyl]);
    this.sampleLoops.push(new SampleLoop('vinyl', randomVinyl, '0:0', `${this.numMeasures}:0`));
  }

  produceIntro() { }

  produceMain() {
    // drumbeat
    this.samples.push(['drumloop1', 0]);
    this.sampleLoops.push(new SampleLoop('drumloop1', 0, '0:0', `${this.numMeasures}:0`));

    this.instruments.push('guitar-bass', 'piano', 'guitar-electric');
    for (let i = 0; i < this.numberOfIterations; i += 1) {
      for (let chordNo = 0; chordNo < this.chordProgression.length; chordNo += 1) {
        const measure = i * this.chordProgression.length + chordNo;
        const chordIndex = this.chordProgression[chordNo] - 1;
        const chordString = this.chordsInScale[chordIndex];
        // e.g. Chord.getChord("maj7", "G4")
        const chord = Chord.getChord(Chord.get(chordString).aliases[0], `${this.notesInScale[chordIndex]}3`);

        // bass line: on the first beat of every measure
        const rootNote = Mode.notes(this.mode, `${this.tonic}1`)[chordIndex];
        const bassTiming = new InstrumentNote('guitar-bass', rootNote, '1m', Producer.toTime(measure, 0));
        this.instrumentNotes.push(bassTiming);

        // arpeggiated chords on the second beat
        for (let note = 0; note < 4; note += 1) {
          this.instrumentNotes.push(
            new InstrumentNote(
              i % 2 === 0 ? 'piano' : 'guitar-electric',
              Note.simplify(chord.notes[note]),
              '0:3',
              Producer.toTime(measure, note * 0.25 + 1)
            )
          );
        }
      }
    }
  }

  produceOutro() { }

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
