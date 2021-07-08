import * as Tonal from '@tonaljs/tonal';
import { Time } from 'tone/build/esm/core/type/Units';
import { InstrumentNote, SampleLoop, Track } from './track';
import { OutputParams } from './params';
import {
  addTime,
  Chord,
  keyNumberToString,
  mapNote,
  mountNotesOnScale,
  octShift,
  octShiftAll,
  randomColor,
  randomFromInterval,
  subtractTime
} from './helper';
import { SAMPLEGROUPS, selectDrumbeat } from './samples';
import { Instrument } from './instruments';
import * as Presets from './producer_presets';

/**
 * The producer takes OutputParams to produce a Track.
 * The production process is deterministic, i.e. the same input will always yield the same output.
 */
class Producer {
  tonic: string;

  keyNum: number;

  mode: string;

  modeNum: number;

  /** How energetic the track should be, 0 (less energetic) to 1 (very energetic) */
  energy: number;

  /** How positive the music should be, 0 (sad) to 1 (cheerful) */
  valence: number;

  notesInScale: string[];

  notesInScalePitched: string[];

  chordsInScale: string[];

  chords: number[];

  chordsTonal: Chord[];

  melodies: number[][];

  bpm: number;

  numMeasures: number;

  introLength: number;

  mainLength: number;

  outroLength: number;

  samples: [string, number][] = [];

  sampleLoops: SampleLoop[] = [];

  instruments: Instrument[] = [];

  instrumentNotes: InstrumentNote[] = [];

  drumbeatStartCurr: Time;

  drumbeatTimings: [Time, Time][] = [];

  produce(params: OutputParams): Track {
    // must be 70, 75, 80, 85, 90, 95 or 100
    let bpm = Math.round(params.bpm / 5) * 5;
    if (bpm < 70) bpm = 70;
    if (bpm > 100) bpm = 100;
    this.bpm = bpm;

    // tonic note, e.g. 'G'
    this.tonic = keyNumberToString(params.key);
    this.keyNum = params.key;

    // musical mode, e.g. 'ionian'
    this.mode = Tonal.Mode.names()[params.mode - 1];
    this.simplifyKeySignature();
    this.modeNum = params.mode;

    // array of notes, e.g. ["C", "D", "E", "F", "G", "A", "B"]
    this.notesInScale = Tonal.Mode.notes(this.mode, this.tonic);
    this.notesInScalePitched = Tonal.Mode.notes(this.mode, `${this.tonic}3`);

    // array of seventh chords, e.g. ["C7", "Dm7", "Em7", "F7", "G7", "Am7", "Bdim7"]
    this.chordsInScale = Tonal.Mode.seventhChords(this.mode, this.tonic);

    this.energy = params.energy;
    this.valence = params.valence;
    this.chords = params.chords;

    const swing = randomFromInterval(1, 10, this.energy) <= 1;

    if (this.chords[0] === 0) {
      this.chords[0] = 1;
    } else if (this.chords[0] !== 1) {
      this.chords.unshift(1);
      params.melodies.unshift([0, 0, 0, 0, 0, 0, 0, 0]);
    }

    this.chordsTonal = this.chords.map((c, chordNo) => {
      const chordIndex = this.chords[chordNo] - 1;
      const chordString = this.chordsInScale[chordIndex];
      // e.g. Chord.getChord("maj7", "G4")
      return Tonal.Chord.getChord(
        Tonal.Chord.get(chordString).aliases[0],
        `${this.notesInScale[chordIndex]}3`
      );
    });
    this.melodies = params.melodies;

    this.introLength = this.produceIntro();
    this.mainLength = this.produceMain();
    this.outroLength = this.produceOutro();

    this.numMeasures = this.introLength + this.mainLength + this.outroLength;
    this.produceFx();

    // drumbeat
    const [drumbeatGroup, drumbeatIndex] = selectDrumbeat(this.bpm, this.energy);
    this.drumbeatTimings.forEach(([startTime, endTime]) => {
      this.addSample(drumbeatGroup, drumbeatIndex, `${startTime}:0`, `${endTime}:0`);
    });

    const title = params.title || `Lofi track in ${this.tonic} ${this.mode}`;
    const track = new Track({
      title,
      swing,
      key: this.tonic,
      keyNum: this.keyNum,
      mode: this.mode,
      modeNum: this.modeNum,
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
    // const length = 1 + Math.ceil(this.chordProgressionChords.length / 4) + 1;
    // this.chordProgressionChords.forEach((chord, chordNo) => {
    //   // hold the last arpeggio longer
    //   const duration = chordNo === this.chordProgression.length - 1 ? '1:1' : '0:2';
    //   this.addArpeggio(Instrument.ElectricGuitar, chord.notes, duration, '64n', `1:${chordNo}`);
    // });
    // return length;

    // const end = this.addOneShot(0);
    return 1;
  }

  produceMain(): number {
    const numberOfIterations = Math.ceil(36 / this.chords.length);
    const length = this.chords.length * numberOfIterations;

    // the measure where the main part starts
    const measureStart = this.introLength;

    const preset = Presets.selectPreset(this.valence, this.energy);

    for (let i = 0; i < numberOfIterations; i += 1) {
      const iterationMeasure = measureStart + i * this.chords.length;
      this.startDrumbeat(`${i === 0 ? iterationMeasure + 2 : iterationMeasure}:0`);
      this.endDrumbeat(`${measureStart + (i + 1) * this.chords.length - 2}:0`);

      this.chords.forEach((scaleDegree, chordNo) => {
        const chord = this.chordsTonal[chordNo];
        const measure = iterationMeasure + chordNo;

        if (!chord.empty) {
          // bass line: on the first beat of every measure
          if (preset.bassLine) {
            const rootNote = `${this.notesInScale[scaleDegree - 1]}${
              1 + preset.bassLine.octaveShift
            }`;
            // get a random bass pattern
            const bassPatternNo = randomFromInterval(
              0,
              Presets.BassPatterns.length - 1,
              this.energy + this.valence + iterationMeasure + chordNo
            );
            const bassPattern = Presets.BassPatterns[bassPatternNo];
            bassPattern.forEach(([startBeat, duration]) => {
              this.addNote(
                preset.bassLine.instrument,
                rootNote,
                `0:${duration}`,
                `${measure}:${startBeat}`,
                preset.bassLine.volume
              );
            });
          }

          // harmony
          if (preset.harmony) {
            const harmonyNotes = chord.notes.slice(0, 3);
            harmonyNotes[0] = octShift(harmonyNotes[0], preset.harmony.octaveShift);
            this.addNote(
              preset.harmony.instrument,
              harmonyNotes,
              '1m',
              `${measure}:0`,
              preset.harmony.volume
            );
          }

          // first beat arpeggio
          if (preset.firstBeatArpeggio) {
            const arpeggioNotes = octShiftAll(
              mountNotesOnScale(scaleDegree, [1, 5, 8, 9, 10], this.notesInScalePitched),
              preset.firstBeatArpeggio.octaveShift
            );
            this.addArpeggio(
              preset.firstBeatArpeggio.instrument,
              arpeggioNotes,
              '0:4',
              '8n',
              `${measure}:0`,
              preset.firstBeatArpeggio.volume
            );
          }

          // second beat arpeggio
          if (preset.secondBeatArpeggio) {
            const arpeggioNotes = octShiftAll(chord.notes, preset.secondBeatArpeggio.octaveShift);
            this.addArpeggio(
              preset.secondBeatArpeggio.instrument,
              arpeggioNotes,
              '0:3',
              '16n',
              `${measure}:1`,
              preset.secondBeatArpeggio.volume
            );
          }
        }

        // this should not happen
        if (this.melodies.length === 0 || !preset.melody) {
          return;
        }
        // reduce into array of [note, length, i]
        const notes = this.melodies[chordNo].reduce(
          ([arr, top], curr, i) => {
            if (top.length === 0) {
              return [arr, [curr, 1, i]];
            }
            if (top[0] === curr) {
              return [arr, [curr, top[1] + 1, top[2]]];
            }
            return [
              [...arr, top],
              [curr, 1, i]
            ];
          },
          [[], []]
        );
        const notesReduced = [...notes[0], notes[1]];
        notesReduced.forEach(([note, length, i]) => {
          const [scaleDegreeIndex, octave] = mapNote(note);

          if (scaleDegreeIndex >= 0) {
            const n = octShift(
              this.notesInScalePitched[scaleDegreeIndex],
              octave + preset.melody.octaveShift
            );
            const melody = preset.melodyOctaves ? [octShift(n, -1), n] : n;
            this.addNote(
              preset.melody.instrument,
              melody,
              `0:0:${length * 4}`,
              `${measure}:0:${i * 2}`,
              preset.melody.volume
            );
          }
        });
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
      this.addSample('rain', randomRain, '0:0', `${this.numMeasures - 0.5}:0`);
    } else {
      // add vinyl crackle
      const randomVinyl = SAMPLEGROUPS.get('vinyl').getRandomSample(this.valence + this.energy);
      // end half a measure before the end
      this.addSample('vinyl', randomVinyl, '0:0', `${this.numMeasures - 0.5}:0`);
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

  startDrumbeat(time: Time) {
    this.drumbeatStartCurr = time;
  }

  endDrumbeat(time: Time) {
    this.drumbeatTimings.push([this.drumbeatStartCurr, time]);
    this.drumbeatStartCurr = undefined;
  }

  addSample(sample: string, sampleIndex: number, startTime: Time, stopTime: Time) {
    if (!this.samples.some(([s, i]) => s === sample && i === sampleIndex)) {
      this.samples.push([sample, sampleIndex]);
    }
    this.sampleLoops.push(new SampleLoop(sample, sampleIndex, startTime, stopTime));
  }

  addNote(
    instrument: Instrument,
    pitch: string | string[],
    duration: Time,
    time: Time,
    velocity?: number
  ) {
    if (!this.instruments.some((i) => i === instrument)) {
      this.instruments.push(instrument);
    }
    this.instrumentNotes.push(new InstrumentNote(instrument, pitch, duration, time, velocity));
  }

  /** Adds a rolling arpeggio to the note list */
  addArpeggio(
    instrument: Instrument,
    notes: string[],
    totalDuration: Time,
    singleNoteUnit: string,
    startTime: Time,
    velocity?: number
  ) {
    notes.forEach((note, i) => {
      const noteDuration = {} as any;
      noteDuration[singleNoteUnit] = i;
      this.addNote(
        instrument,
        note,
        subtractTime(totalDuration, noteDuration),
        addTime(startTime, noteDuration),
        velocity
      );
    });
  }

  /** Adds a oneshot, centered within a measure window */
  // addOneShot(startMeasure: number) {
  //   const oneshotGroup = SAMPLEGROUPS.get('guitar-electric-major');
  //   const oneshotSample = oneshotGroup.getRandomSampleByKey(
  //     this.energy + this.valence,
  //     this.keyNum
  //   );

  //   // get length in seconds
  //   const oneshotDuration = oneshotGroup.durations[oneshotSample];

  //   // get length in measures
  //   const measureDuration = 60 / (this.bpm / 4);
  //   const oneshotLength = oneshotDuration / measureDuration;
  //   const oneshotMeasures = Math.ceil(oneshotLength);
  //   const offsetMeasure = oneshotMeasures - oneshotLength;

  //   // return endMeasure
  //   this.addSample(oneshotGroup.name, oneshotSample, `${startMeasure}:0`, `${startMeasure + oneshotMeasures - offsetMeasure}:0`);
  //   return oneshotMeasures;
  // }
}

export default Producer;
