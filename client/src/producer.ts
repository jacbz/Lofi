import { Chord, Key, Mode, Note, Scale } from '@tonaljs/tonal';
import { Time } from 'tone/build/esm/core/type/Units';
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

  numMeasures: number;

  introLength: number;

  mainLength: number;

  outroLength: number;

  samples: [string, number][] = [];

  sampleLoops: SampleLoop[] = [];

  instruments: string[] = [];

  instrumentNotes: InstrumentNote[] = [];

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

    this.introLength = this.produceIntro();
    this.mainLength = this.produceMain();
    this.outroLength = this.produceOutro();

    this.numMeasures = this.introLength + this.mainLength + this.outroLength;
    this.produceFx();

    const bpm = 70;
    const title = `Lofi track in ${this.tonic} ${this.mode}`;
    const track = new Track({
      title,
      mode: this.mode,
      key: this.tonic,
      energy: this.energy,
      valence: this.valence,
      numMeasures: this.numMeasures,
      bpm,
      samples: this.samples,
      sampleLoops: this.sampleLoops,
      instruments: this.instruments,
      instrumentNotes: this.instrumentNotes
    });
    return track;
  }

  addNote(instrument: string, pitch: string | string[], duration: Time, time: Time) {
    if (!this.instruments.some((i) => i === instrument)) {
      this.instruments.push(instrument);
    }
    this.instrumentNotes.push(new InstrumentNote(instrument, pitch, duration, time));
  }

  addLoop(sample: string, sampleIndex: number, startTime: Time, stopTime: Time) {
    if (!this.samples.some(([s, i]) => s === sample && i === sampleIndex)) {
      this.samples.push([sample, sampleIndex]);
    }
    this.sampleLoops.push(new SampleLoop(sample, sampleIndex, startTime, stopTime));
  }

  produceIntro(): number {
    const measureEnd = 0;
    // silent intro (except fx)
    return measureEnd;
  }

  produceMain(): number {
    const numberOfIterations = 6;
    const length = this.chordProgression.length * numberOfIterations;

    // measure where the main part starts
    const measureStart = this.introLength;
    // measure where the main part ends
    const measureEnd = this.introLength + length;

    // drumbeat
    this.addLoop('drumloop2', 0, `${measureStart}:0`, `${measureEnd}:0`);

    for (let i = 0; i < numberOfIterations; i += 1) {
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
        this.addNote('guitar-bass', rootNote, '1m', `${measure}:0`);

        // arpeggiated chords on the second beat
        for (let note = 0; note < 4; note += 1) {
          const instrument = i % 2 === 0 ? 'piano' : 'guitar-electric';
          this.addNote(instrument, chord.notes[note], '0:3', `${measure}:${note * 0.25 + 1}`);
        }
      }
    }

    return length;
  }

  produceOutro(): number {
    // measure where the outro part starts
    const measureStart = this.introLength + this.mainLength;
    // add an empty measure of silence at the end
    const length = 2;

    // end with I9 chord
    const i9chord = Chord.getChord('9', `${this.tonic}2`);
    for (let note = 0; note < i9chord.notes.length; note += 1) {
      this.addNote('piano', i9chord.notes[note], '0:3', `${measureStart}:${note * 0.25}`);
    }

    // ending bass note
    this.addNote('guitar-bass', `${this.tonic}1`, '1m', `${measureStart}:${0}`);

    // leading tone for resolution
    const resolutionNoteTime = `${measureStart - 1}:${3}`;
    this.addNote('piano', Note.transpose(`${this.tonic}2`, '-2M'), '4n', resolutionNoteTime);
    this.addNote('guitar-bass', Note.transpose(`${this.tonic}1`, '-2M'), '4n', resolutionNoteTime);

    return length;
  }

  produceFx() {
    // vinyl crackle
    const randomVinyl = Producer.randomFromInterval(
      0,
      LOOPS.get('vinyl').size - 1,
      this.energy + this.valence
    );
    // end half a measure before the end
    this.addLoop('vinyl', randomVinyl, '0:0', `${this.numMeasures - 0.5}:0`);
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
