import * as Tonal from '@tonaljs/tonal';
import { Time } from 'tone/build/esm/core/type/Units';
import { InstrumentNote, SampleLoop, Track } from './track';
import { OutputParams } from './params';
import { addTime, Chord, octShift, randomColor, subtractTime } from './helper';
import { SAMPLEGROUPS, selectDrumbeat } from './samples';
import { Instrument } from './instruments';

/**
 * The producer takes OutputParams to produce a Track.
 * The production process is deterministic, i.e. the same input will always yield the same output.
 */
class Producer {
  tonic: string;

  mode: string;

  /** How energetic the track should be, 0 (less energetic) to 1 (very energetic) */
  energy: number;

  /** How positive the music should be, 0 (sad) to 1 (cheerful) */
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

  instruments: Instrument[] = [];

  instrumentNotes: InstrumentNote[] = [];

  produce(params: OutputParams): Track {
    // must be 70, 75, 80, 85, 90, 95 or 100
    let bpm = Math.round(params.bpm / 5) * 5;
    if (bpm < 70) bpm = 70;
    if (bpm > 100) bpm = 100;
    this.bpm = bpm;

    // tonic note, e.g. 'G'
    this.tonic = Tonal.Scale.get('C chromatic').notes[params.key - 1];

    // musical mode, e.g. 'ionian'
    this.mode = Tonal.Mode.names()[params.mode - 1];
    this.simplifyKeySignature();

    // array of notes, e.g. ["C", "D", "E", "F", "G", "A", "B"]
    this.notesInScale = Tonal.Mode.notes(this.mode, this.tonic);

    // array of seventh chords, e.g. ["C7", "Dm7", "Em7", "F7", "G7", "Am7", "Bdim7"]
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
      numMeasures: this.numMeasures,
      bpm: this.bpm,
      samples: this.samples,
      sampleLoops: this.sampleLoops,
      instruments: this.instruments,
      instrumentNotes: this.instrumentNotes,
      color: randomColor(this.energy + this.valence),
      outputParams: params
    });
    return track;
  }

  produceIntro(): number {
    // one empty measure, arpeggios, followed by one empty measure
    const length = 1 + Math.ceil(this.chordProgressionChords.length / 4) + 1;
    this.chordProgressionChords.forEach((chord, chordNo) => {
      // hold the last arpeggio longer
      const duration = chordNo === this.chordProgression.length - 1 ? '1:1' : '0:2';
      this.addArpeggio(Instrument.ElectricGuitar, chord.notes, duration, '64n', `1:${chordNo}`);
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
        this.addNote(Instrument.BassGuitar, rootNote, '1m', `${measure}:0`);

        // arpeggiated chords on the second beat
        this.addArpeggio(Instrument.Piano, chord.notes, '0:3', '16n', `${measure}:1`);
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
    this.addNote(Instrument.Piano, resolutionNote, '4n', resolutionNoteTime);
    this.addNote(Instrument.BassGuitar, octShift(resolutionNote, -1), '4n', resolutionNoteTime);

    // end with I9 chord
    const i9chord = Tonal.Chord.getChord('9', `${this.tonic}2`);
    this.addArpeggio(Instrument.Piano, i9chord.notes, '1:2', '16n', `${measureStart}:0`);

    // ending bass note
    this.addNote(Instrument.BassGuitar, `${this.tonic}1`, '1m', `${measureStart}:0`);

    return length;
  }

  produceFx() {
    if (this.valence < 0.2) {
      // add rain
      const randomRain = SAMPLEGROUPS.get('rain').getRandomSample(this.valence);
      // end half a measure before the end
      this.addLoop('rain', randomRain, '0:0', `${this.numMeasures - 0.5}:0`);
    } else {
      // add vinyl crackle
      const randomVinyl = SAMPLEGROUPS.get('vinyl').getRandomSample(this.valence + this.energy);
      // end half a measure before the end
      this.addLoop('vinyl', randomVinyl, '0:0', `${this.numMeasures - 0.5}:0`);
    }
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

  addNote(instrument: Instrument, pitch: string | string[], duration: Time, time: Time) {
    if (!this.instruments.some((i) => i === instrument)) {
      this.instruments.push(instrument);
    }
    this.instrumentNotes.push(new InstrumentNote(instrument, pitch, duration, time));
  }

  /** Adds a rolling arpeggio to the note list */
  addArpeggio(
    instrument: Instrument,
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
}

export default Producer;
