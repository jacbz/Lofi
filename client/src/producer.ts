import * as Tone from 'tone';
import * as Tonal from '@tonaljs/tonal';
import { Time } from 'tone/build/esm/core/type/Units';
import { InstrumentNote, SampleLoop, Track } from './track';
import { OutputParams } from './params';
import { LOOPS, selectDrumbeat } from './samples';
import { addTime, Chord, octShift, subtractTime } from './music';

class Producer {
  tonic: string;

  mode: string;

  energy: number;

  valence: number;

  notesInScale: string[];

  chordsInScale: string[];

  chordProgression: number[];

  chordProgressionChords: Chord[];

  bpm: number;

  numMeasures: number;

  introLength: number;

  mainLength: number;

  outroLength: number;

  samples: [string, number][] = [];

  sampleLoops: SampleLoop[] = [];

  instruments: string[] = [];

  instrumentNotes: InstrumentNote[] = [];

  produce(params: OutputParams): Track {
    // must be 70, 75, 80, 85, 90, 95 or 100
    this.bpm = 75;

    // tonic note, e.g. 'G'
    this.tonic = Tonal.Scale.get('C chromatic').notes[params.key - 1];

    // musical mode, e.g. 'ionian'
    this.mode = Tonal.Mode.names()[params.mode - 1];

    this.simplifyKeySignature();

    // array of notes, e.g. ["C", "D", "E", "F", "G", "A", "B"]
    this.notesInScale = Tonal.Mode.notes(this.mode, this.tonic);

    // array of triads, e.g. ["C", "Dm", "Em", "F", "G", "Am", "Bdim"]
    this.chordsInScale = Tonal.Mode.seventhChords(this.mode, this.tonic);

    this.energy = params.energy;
    this.valence = params.valence;
    this.chordProgression = params.chordProgression;
    this.chordProgressionChords = this.chordProgression.map((c, chordNo) => {
      const chordIndex = this.chordProgression[chordNo] - 1;
      const chordString = this.chordsInScale[chordIndex];
      // e.g. Chord.getChord("maj7", "G4")
      return Tonal.Chord.getChord(
        Tonal.Chord.get(chordString).aliases[0],
        `${this.notesInScale[chordIndex]}3`
      );
    });
    console.log(this.chordProgressionChords);

    this.introLength = this.produceIntro();
    this.mainLength = this.produceMain();
    this.outroLength = this.produceOutro();

    this.numMeasures = this.introLength + this.mainLength + this.outroLength;
    this.produceFx();

    const title = `Lofi track in ${this.tonic} ${this.mode}`;
    const track = new Track({
      title,
      mode: this.mode,
      key: this.tonic,
      energy: this.energy,
      valence: this.valence,
      numMeasures: this.numMeasures,
      bpm: this.bpm,
      samples: this.samples,
      sampleLoops: this.sampleLoops,
      instruments: this.instruments,
      instrumentNotes: this.instrumentNotes
    });
    return track;
  }

  produceIntro(): number {
    // one empty measure, arpeggios, followed by one empty measure
    const length = 1 + Math.ceil(this.chordProgressionChords.length / 4) + 1;
    this.chordProgressionChords.forEach((chord, chordNo) => {
      // hold the last arpeggio longer
      const duration = chordNo === this.chordProgression.length - 1 ? '1:1' : '0:2';
      this.addArpeggio('guitar-electric', chord.notes, duration, '64n', `1:${chordNo}`);
    });
    return length;
  }

  produceMain(): number {
    const numberOfIterations = 6;
    const length = this.chordProgression.length * numberOfIterations;

    // the measure where the main part starts
    const measureStart = this.introLength;
    // the measure where the main part ends
    const measureEnd = this.introLength + length;

    // drumbeat
    const drumbeat = selectDrumbeat(this.bpm, this.energy);
    this.addLoop(drumbeat[0], drumbeat[1], `${measureStart}:0`, `${measureEnd}:0`);

    for (let i = 0; i < numberOfIterations; i += 1) {
      this.chordProgressionChords.forEach((chord, chordNo) => {
        const measure = measureStart + i * this.chordProgression.length + chordNo;
        // bass line: on the first beat of every measure
        const rootNote = octShift(`${chord.tonic}`, -1);
        this.addNote('guitar-bass', rootNote, '1m', `${measure}:0`);

        // arpeggiated chords on the second beat
        this.addArpeggio('piano', chord.notes, '0:3', '16n', `${measure}:1`);
      });
    }

    return length;
  }

  produceOutro(): number {
    // the measure where the outro part starts
    const measureStart = this.introLength + this.mainLength;
    // add an empty measure of silence at the end
    const length = 2;

    // leading tone for resolution
    const resolutionNoteTime = `${measureStart - 1}:${3}`;
    const resolutionNote = Tonal.Note.transpose(`${this.tonic}2`, '-2M');
    this.addNote('piano', resolutionNote, '4n', resolutionNoteTime);
    this.addNote('guitar-bass', octShift(resolutionNote, -1), '4n', resolutionNoteTime);

    // end with I9 chord
    const i9chord = Tonal.Chord.getChord('9', `${this.tonic}2`);
    this.addArpeggio('piano', i9chord.notes, '1:2', '16n', `${measureStart}:0`);

    // ending bass note
    this.addNote('guitar-bass', `${this.tonic}1`, '1m', `${measureStart}:0`);

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
      const enharmonic = Tonal.Note.enharmonic(this.tonic);
      const enharmonicKey = Tonal.Key.majorKey(enharmonic);
      if (Tonal.Key.majorKey(this.tonic).keySignature.length >= enharmonicKey.keySignature.length) {
        this.tonic = enharmonic;
      }
    }
    if (this.mode === 'aeolian') {
      this.mode = 'minor';
      const enharmonic = Tonal.Note.enharmonic(this.tonic);
      const enharmonicKey = Tonal.Key.majorKey(enharmonic);
      if (Tonal.Key.minorKey(this.tonic).keySignature.length >= enharmonicKey.keySignature.length) {
        this.tonic = enharmonic;
      }
    }
  }

  addLoop(sample: string, sampleIndex: number, startTime: Time, stopTime: Time) {
    if (!this.samples.some(([s, i]) => s === sample && i === sampleIndex)) {
      this.samples.push([sample, sampleIndex]);
    }
    this.sampleLoops.push(new SampleLoop(sample, sampleIndex, startTime, stopTime));
  }

  addNote(instrument: string, pitch: string | string[], duration: Time, time: Time) {
    if (!this.instruments.some((i) => i === instrument)) {
      this.instruments.push(instrument);
    }
    this.instrumentNotes.push(new InstrumentNote(instrument, pitch, duration, time));
  }

  /** Adds a rolling arpeggio to the note list */
  addArpeggio(
    instrument: string,
    notes: string[],
    totalDuration: Time,
    singleNoteUnit: string,
    startTime: Time
  ) {
    notes.forEach((note, i) => {
      const noteDuration = {} as any;
      noteDuration[singleNoteUnit] = i;
      this.addNote(
        instrument,
        note,
        subtractTime(totalDuration, noteDuration),
        addTime(startTime, noteDuration)
      );
    });
  }

  /** Returns a quasi-random number between min-max based on given seed number */
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
