import * as Tone from 'tone';
import { SAMPLES_BASE_URL } from './samples';

export enum Instrument {
  /** Salamander grand piano, velocity 6 */
  Piano,
  /** Salamander grand piano, velocity 1 */
  SoftPiano,
  /** Mellow electric piano */
  ElectricPiano,
  /** Harp */
  Harp,
  /** Acoustic guitar */
  AcousticGuitar,
  /** Bass Guitar */
  BassGuitar,
  /** Electric guitar */
  ElectricGuitar,
  /** Synth (Tone.js) */
  Synth
}

const BASE_URL = `${SAMPLES_BASE_URL}/instruments`;
export const getInstrument = (instrument: Instrument) => {
  switch (instrument) {
    case Instrument.Piano: {
      return new Tone.Sampler({
        urls: {
          A2: 'A2.mp3',
          A3: 'A3.mp3',
          A4: 'A4.mp3',
          A5: 'A5.mp3',
          A6: 'A6.mp3',
          A7: 'A7.mp3',
          C2: 'C2.mp3',
          C3: 'C3.mp3',
          C4: 'C4.mp3',
          C5: 'C5.mp3',
          C6: 'C6.mp3',
          C7: 'C7.mp3',
          C8: 'C8.mp3',
          'D#2': 'Ds2.mp3',
          'D#3': 'Ds3.mp3',
          'D#4': 'Ds4.mp3',
          'D#5': 'Ds5.mp3',
          'D#6': 'Ds6.mp3',
          'D#7': 'Ds7.mp3',
          'F#2': 'Fs2.mp3',
          'F#3': 'Fs3.mp3',
          'F#4': 'Fs4.mp3',
          'F#5': 'Fs5.mp3',
          'F#6': 'Fs6.mp3',
          'F#7': 'Fs7.mp3'
        },
        baseUrl: `${BASE_URL}/piano/`,
        volume: 0
      });
    }

    case Instrument.SoftPiano: {
      return new Tone.Sampler({
        urls: {
          A2: 'A2.mp3',
          A3: 'A3.mp3',
          A4: 'A4.mp3',
          A5: 'A5.mp3',
          A6: 'A6.mp3',
          A7: 'A7.mp3',
          C2: 'C2.mp3',
          C3: 'C3.mp3',
          C4: 'C4.mp3',
          C5: 'C5.mp3',
          C6: 'C6.mp3',
          C7: 'C7.mp3',
          C8: 'C8.mp3',
          'D#2': 'Ds2.mp3',
          'D#3': 'Ds3.mp3',
          'D#4': 'Ds4.mp3',
          'D#5': 'Ds5.mp3',
          'D#6': 'Ds6.mp3',
          'D#7': 'Ds7.mp3',
          'F#2': 'Fs2.mp3',
          'F#3': 'Fs3.mp3',
          'F#4': 'Fs4.mp3',
          'F#5': 'Fs5.mp3',
          'F#6': 'Fs6.mp3',
          'F#7': 'Fs7.mp3'
        },
        baseUrl: `${BASE_URL}/piano-soft/`,
        volume: 4
      });
    }

    case Instrument.ElectricPiano: {
      return new Tone.Sampler({
        urls: {
          A2: 'A2.mp3',
          A3: 'A3.mp3',
          A4: 'A4.mp3',
          A5: 'A5.mp3',
          A6: 'A6.mp3',
          C2: 'C2.mp3',
          C3: 'C3.mp3',
          C4: 'C4.mp3',
          C5: 'C5.mp3',
          C6: 'C6.mp3',
          'D#2': 'Ds2.mp3',
          'D#3': 'Ds3.mp3',
          'D#4': 'Ds4.mp3',
          'D#5': 'Ds5.mp3',
          'D#6': 'Ds6.mp3',
          'F#2': 'Fs2.mp3',
          'F#3': 'Fs3.mp3',
          'F#4': 'Fs4.mp3',
          'F#5': 'Fs5.mp3',
          'F#6': 'Fs6.mp3'
        },
        baseUrl: `${BASE_URL}/piano-electric/`,
        volume: 0
      });
    }

    case Instrument.Harp: {
      return new Tone.Sampler({
        urls: {
          A2: 'A2.mp3',
          A3: 'A3.mp3',
          A4: 'A4.mp3',
          A5: 'A5.mp3',
          A6: 'A6.mp3',
          C2: 'C2.mp3',
          C3: 'C3.mp3',
          C4: 'C4.mp3',
          C5: 'C5.mp3',
          C6: 'C6.mp3',
          'D#2': 'Ds2.mp3',
          'D#3': 'Ds3.mp3',
          'D#4': 'Ds4.mp3',
          'D#5': 'Ds5.mp3',
          'D#6': 'Ds6.mp3',
          'F#2': 'Fs2.mp3',
          'F#3': 'Fs3.mp3',
          'F#4': 'Fs4.mp3',
          'F#5': 'Fs5.mp3',
          'F#6': 'Fs6.mp3'
        },
        baseUrl: `${BASE_URL}/harp/`,
        volume: 0
      });
    }

    case Instrument.AcousticGuitar: {
      return new Tone.Sampler({
        urls: {
          A2: 'A2.mp3',
          A3: 'A3.mp3',
          A4: 'A4.mp3',
          A5: 'A5.mp3',
          C2: 'C2.mp3',
          C3: 'C3.mp3',
          C4: 'C4.mp3',
          C5: 'C5.mp3',
          C6: 'C6.mp3',
          'D#2': 'Ds2.mp3',
          'D#3': 'Ds3.mp3',
          'D#4': 'Ds4.mp3',
          'D#5': 'Ds5.mp3',
          'F#2': 'Fs2.mp3',
          'F#3': 'Fs3.mp3',
          'F#4': 'Fs4.mp3',
          'F#5': 'Fs5.mp3'
        },
        baseUrl: `${BASE_URL}/guitar-acoustic/`,
        volume: 0
      });
    }

    case Instrument.ElectricGuitar: {
      return new Tone.Sampler({
        urls: {
          A2: 'A2.mp3',
          A3: 'A3.mp3',
          A4: 'A4.mp3',
          A5: 'A5.mp3',
          C3: 'C3.mp3',
          C4: 'C4.mp3',
          C5: 'C5.mp3',
          C6: 'C6.mp3',
          'C#2': 'Cs2.mp3',
          'D#3': 'Ds3.mp3',
          'D#4': 'Ds4.mp3',
          'D#5': 'Ds5.mp3',
          E2: 'E2.mp3',
          'F#2': 'Fs2.mp3',
          'F#3': 'Fs3.mp3',
          'F#4': 'Fs4.mp3',
          'F#5': 'Fs5.mp3'
        },
        baseUrl: `${BASE_URL}/guitar-electric/`,
        volume: -10
      });
    }

    case Instrument.BassGuitar: {
      return new Tone.Sampler({
        urls: {
          E1: 'E.mp3',
          A1: 'A.mp3',
          C2: 'C.mp3'
        },
        baseUrl: `${BASE_URL}/guitar-bass/`,
        volume: 0
      });
    }

    case Instrument.Synth: {
      return new Tone.PolySynth(Tone.Synth, {
        envelope: {
          attack: 0.02,
          decay: 0.1,
          sustain: 0.3,
          release: 1
        },
        volume: -10
      });
    }

    default:
      throw new Error('Invalid instrument specified');
  }
};

export const DefaultFilters = [
  new Tone.Reverb({
    decay: 2,
    wet: 0.2,
    preDelay: 0.3
  })
];

export const getInstrumentFilters = (instrument: Instrument) => {
  switch (instrument) {
    case Instrument.ElectricGuitar: {
      return [
        ...DefaultFilters,
        new Tone.Filter({
          type: 'highpass',
          frequency: 350,
          Q: 0.5
        })
      ];
    }

    case Instrument.BassGuitar: {
      return [
        ...DefaultFilters,
        new Tone.Filter({
          type: 'highpass',
          frequency: 300,
          Q: 0.5
        })
      ];
    }

    default:
      return [
        ...DefaultFilters
      ];
  }
};
