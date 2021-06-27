import { Chord, Key, Mode, Note, Scale } from '@tonaljs/tonal';
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

  introLength: number;

  numberOfIterations = 6;

  numMeasures: number;

  mainLength: number;

  outroLength: number;

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

    this.simplifyKeySignature();

    // array of notes, e.g. ["C", "D", "E", "F", "G", "A", "B"]
    this.notesInScale = Mode.notes(this.mode, this.tonic);

    // array of triads, e.g. ["C", "Dm", "Em", "F", "G", "Am", "Bdim"]
    this.chordsInScale = Mode.seventhChords(this.mode, this.tonic);
    console.log(this.chordsInScale);

    this.energy = params.energy;
    this.valence = params.valence;
    this.chordProgression = params.chordProgression;

    this.introLength = 1;
    this.numberOfIterations = 6;
    this.mainLength = params.chordProgression.length * this.numberOfIterations;
    this.outroLength = 1;

    this.numMeasures = this.introLength + this.mainLength + this.outroLength;

    this.instruments.push('guitar-bass', 'piano', 'guitar-electric');
    this.produceFx();
    this.produceIntro();
    this.produceMain();
    this.produceOutro();

    const title = `Lofi track in ${this.tonic} ${this.mode}`;
    const track = new Track({
      title,
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
    const randomVinyl = Producer.randomFromInterval(
      0,
      LOOPS.get('vinyl').size - 1,
      this.energy + this.valence
    );
    this.samples.push(['vinyl', randomVinyl]);
    this.sampleLoops.push(new SampleLoop('vinyl', randomVinyl, '0:0', `${this.numMeasures}:0`));
  }

  produceIntro() {
    const measureEnd = this.introLength;
    // silent intro (except fx)
  }

  produceMain() {
    // measure where the main part starts
    const measureStart = this.introLength;
    // measure where the main part ends
    const measureEnd = this.introLength + this.mainLength;

    // drumbeat
    this.samples.push(['drumloop1', 0]);
    this.sampleLoops.push(new SampleLoop('drumloop1', 0, `${measureStart}:0`, `${measureEnd}:0`));

    for (let i = 0; i < this.numberOfIterations; i += 1) {
      for (let chordNo = 0; chordNo < this.chordProgression.length; chordNo += 1) {
        const measure = measureStart + i * this.chordProgression.length + chordNo;
        const chordIndex = this.chordProgression[chordNo] - 1;
        const chordString = this.chordsInScale[chordIndex];
        // e.g. Chord.getChord("maj7", "G4")
        const chord = Chord.getChord(
          Chord.get(chordString).aliases[0],
          `${this.notesInScale[chordIndex]}3`
        );

        // bass line: on the first beat of every measure
        const rootNote = Mode.notes(this.mode, `${this.tonic}1`)[chordIndex];
        const bassTiming = new InstrumentNote(
          'guitar-bass',
          rootNote,
          '1m',
          Producer.toTime(measure, 0)
        );
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

  produceOutro() {
    // measure where the outro part starts
    const measureStart = this.introLength + this.mainLength;

    // end with I9 chord
    const i9chord = Chord.getChord('9', `${this.tonic}2`);
    for (let note = 0; note < i9chord.notes.length; note += 1) {
      this.instrumentNotes.push(
        new InstrumentNote(
          'piano',
          Note.simplify(i9chord.notes[note]),
          '1:0',
          Producer.toTime(measureStart, note * 0.25)
        )
      );
    }

    // ending bass note
    this.instrumentNotes.push(
      new InstrumentNote('guitar-bass', `${this.tonic}1`, '1m', Producer.toTime(measureStart, 0))
    );

    // leading tone for resolution
    this.instrumentNotes.push(
      new InstrumentNote(
        'piano',
        Note.simplify(Note.transpose(`${this.tonic}2`, '-2M')),
        '4n',
        Producer.toTime(measureStart - 1, 3)
      )
    );
    this.instrumentNotes.push(
      new InstrumentNote(
        'guitar-bass',
        Note.simplify(Note.transpose(`${this.tonic}1`, '-2M')),
        '4n',
        Producer.toTime(measureStart - 1, 3)
      )
    );
  }

  /** simplify key signature, e.g. Db major instead of C# major */
  simplifyKeySignature() {
    if (this.mode === 'ionian') {
      this.mode = 'major';
      const enharmonic = Note.enharmonic(this.tonic);
      const enharmonicKey = Key.majorKey(enharmonic);
      if (Key.majorKey(this.tonic).keySignature.length >= enharmonicKey.keySignature.length) {
        this.tonic = enharmonic;
      }
    }
    if (this.mode === 'aeolian') {
      this.mode = 'minor';
      const enharmonic = Note.enharmonic(this.tonic);
      const enharmonicKey = Key.majorKey(enharmonic);
      if (Key.minorKey(this.tonic).keySignature.length >= enharmonicKey.keySignature.length) {
        this.tonic = enharmonic;
      }
    }
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
