import * as Tonal from '@tonaljs/tonal';
import * as Tone from 'tone';
import { Time } from 'tone/build/esm/core/type/Units';
import { InstrumentNote, SampleLoop, Track } from './track';
import { OutputParams } from './params';
import {
  addTime,
  Chord,
  keyNumberToString,
  mapNote,
  measuresToSeconds,
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

  preset: Presets.ProducerPreset;

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

  /** Drum beat timings, as tuples of (isStart, Time) */
  drumbeatTimings: [boolean, Time][] = [];

  /** Takes OutputParams and deterministically produces a Track */
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

    // array of triads, e.g. ["C", "Dm", "Em", "F", "G", "Am", "Bdim"]
    this.chordsInScale = Tonal.Mode.triads(this.mode, this.tonic);

    this.energy = params.energy;
    this.valence = params.valence;
    this.chords = params.chords;

    this.preset = Presets.selectPreset(this.valence, this.energy);

    // swing with probability 1/10
    const swing = randomFromInterval(1, 10, this.energy) <= 1;

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
    this.drumbeatTimings.sort(
      ([_, time], [__, time2]) => Tone.Time(time).toSeconds() - Tone.Time(time2).toSeconds()
    );
    let currentStartTime: Time = null;
    this.drumbeatTimings.forEach(([isStart, time]) => {
      if (isStart) {
        if (!currentStartTime) {
          currentStartTime = time;
        }
      } else if (currentStartTime) {
        this.addSample(drumbeatGroup, drumbeatIndex, `${currentStartTime}:0`, `${time}:0`);
        currentStartTime = null;
      }
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

  /** Produces the track's intro and returns the number of measures */
  produceIntro(): number {
    // TODO: produce a more interesting intro

    // measure of silence
    return 1;
  }

  /** Produces the track's main part and returns the number of measures */
  produceMain(): number {
    const numberOfIterations = Math.ceil(24 / this.chords.length);
    const length = this.chords.length * numberOfIterations;

    // the measure where the main part starts
    const measureStart = this.introLength;

    // number of bars at the beginning and end without a drumbeat
    const drumbeatPadding = this.chords.length > 8 ? 2 : 1;

    for (let i = 0; i < numberOfIterations; i += 1) {
      const iterationMeasure = measureStart + i * this.chords.length;
      this.startDrumbeat(`${i === 0 ? iterationMeasure + drumbeatPadding : iterationMeasure}:0`);
      this.endDrumbeat(`${measureStart + (i + 1) * this.chords.length - drumbeatPadding}:0`);

      this.produceIteration(iterationMeasure);
    }

    return length;
  }

  /** Produces the track's outro and returns the number of measures */
  produceOutro(): number {
    // the measure where the outro part starts
    const measureStart = this.introLength + this.mainLength;

    // play first two chords (for fade out)
    const measures = this.produceIteration(measureStart, 2);
    this.outroLength = measuresToSeconds(measures, this.bpm) * 2;

    // add an empty measure of silence at the end
    const length = measures + 1;

    return length;
  }

  /** Produces FX for the whole track */
  produceFx() {
    if (this.valence < 0.5 && this.modeNum === 6) {
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

  /** Produces a single iteration of the chord progression; can be cut off prematurely */
  produceIteration(iterationMeasure: number, cutoff?: number) {
    let noDrumBeatCurrently = false;
    const chords = cutoff ? this.chords.slice(0, cutoff) : this.chords;
    chords.forEach((scaleDegree, chordNo) => {
      const chord = this.chordsTonal[chordNo];
      const measure = iterationMeasure + chordNo;

      if (chord.empty) {
        this.endDrumbeat(`${measure}:3`);
        noDrumBeatCurrently = true;
      } else {
        if (noDrumBeatCurrently) {
          this.startDrumbeat(`${measure}:0`);
          noDrumBeatCurrently = false;
        }
        // bass line: on the first beat of every measure
        if (this.preset.bassLine) {
          const rootNote = `${this.notesInScale[scaleDegree - 1]}${
            1 + this.preset.bassLine.octaveShift
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
              this.preset.bassLine.instrument,
              rootNote,
              `0:${duration}`,
              `${measure}:${startBeat}`,
              this.preset.bassLine.volume
            );
          });
        }

        // harmony
        if (this.preset.harmony) {
          const harmonyNotes = octShiftAll(chord.notes, this.preset.harmony.octaveShift);
          // invert chord
          harmonyNotes[0] = octShift(harmonyNotes[0], 1);
          this.addNote(
            this.preset.harmony.instrument,
            harmonyNotes,
            '1m',
            `${measure}:0`,
            this.preset.harmony.volume
          );
        }

        // first beat arpeggio
        if (this.preset.firstBeatArpeggio) {
          const arpeggioNotes = octShiftAll(
            mountNotesOnScale(
              scaleDegree,
              this.preset.firstBeatArpeggioPattern,
              this.notesInScalePitched
            ),
            this.preset.firstBeatArpeggio.octaveShift
          );
          this.addArpeggio(
            this.preset.firstBeatArpeggio.instrument,
            arpeggioNotes,
            '0:4',
            '8n',
            `${measure}:0`,
            this.preset.firstBeatArpeggio.volume
          );
        }
      }

      // this should not happen
      if (this.melodies.length === 0 || !this.preset.melody) {
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
            octave + this.preset.melody.octaveShift
          );
          const melody = this.preset.melodyOctaves ? [octShift(n, -1), n] : n;
          this.addNote(
            this.preset.melody.instrument,
            melody,
            null,
            `${measure}:0:${i * 2}`,
            this.preset.melody.volume
          );
        }
      });
    });

    return chords.length;
  }

  /** Simplifies key signature, e.g. Db major instead of C# major */
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

  /** Starts the drumbeat at the given time */
  startDrumbeat(time: Time) {
    this.drumbeatTimings.push([true, time]);
  }

  /** Stops the drumbeat at the given time */
  endDrumbeat(time: Time) {
    this.drumbeatTimings.push([false, time]);
  }

  /** Adds a given sample to the track */
  addSample(sample: string, sampleIndex: number, startTime: Time, stopTime: Time) {
    if (!this.samples.some(([s, i]) => s === sample && i === sampleIndex)) {
      this.samples.push([sample, sampleIndex]);
    }
    this.sampleLoops.push(new SampleLoop(sample, sampleIndex, startTime, stopTime));
  }

  /** Adds a given instrument note to the track */
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
}

export default Producer;
