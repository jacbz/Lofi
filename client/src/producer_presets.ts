import { Instrument } from './instruments';

/**
 * A ProducerPreset defines which parts are being played, which instruments are playing them,
 * and other parameters,
 */
export class ProducerPreset {
  /** Instrument that plays the root note at the first beat beginning of each chord */
  bassLine: InstrumentConfiguration;

  /** Instrument that plays the entire chord at the first beat of each chord */
  harmony: InstrumentConfiguration;

  /** Instrument that plays an arpeggio at the first beat */
  firstBeatArpeggio: InstrumentConfiguration;

  /** First beat arpeggio pattern, played in eighth notes */
  firstBeatArpeggioPattern = [1, 5, 8, 9, 10];

  /** Instrument that plays the melody */
  melody: InstrumentConfiguration;

  /** Play the melody in octaves */
  melodyOctaves = false;

  public constructor(init?: Partial<ProducerPreset>) {
    Object.assign(this, init);
  }
}

/** Defines a particular instrument configuration */
export class InstrumentConfiguration {
  instrument: Instrument;

  /** Volume between [0, 1] */
  volume = 1;

  /** Octaves to shift up or down */
  octaveShift = 0;

  public constructor(init?: Partial<InstrumentConfiguration>) {
    Object.assign(this, init);
  }
}

/** Deterministically selects a specific ProducerPreset based on valence and energy */
export const selectPreset = (valence: number, energy: number): ProducerPreset => {
  if (energy < 0.3 && valence < 0.5) {
    return Preset3;
  }
  if (energy < 0.6) {
    return Preset2;
  }
  return Preset1;
};

/** A preset, with electric guitar melodies */
export const Preset1: ProducerPreset = new ProducerPreset({
  bassLine: new InstrumentConfiguration({
    instrument: Instrument.BassGuitar,
    volume: 0.6
  }),
  harmony: new InstrumentConfiguration({
    instrument: Instrument.ElectricPiano,
    volume: 0.8
  }),
  firstBeatArpeggio: new InstrumentConfiguration({
    instrument: Instrument.Piano,
    octaveShift: -1,
    volume: 0.1
  }),
  firstBeatArpeggioPattern: [1, 5, 8, 5, 10, 5, 8],
  melody: new InstrumentConfiguration({
    instrument: Instrument.ElectricGuitar,
    octaveShift: 0,
    volume: 0.6
  })
});

/** A softer preset, with epiano melodies */
export const Preset2: ProducerPreset = new ProducerPreset({
  bassLine: new InstrumentConfiguration({
    instrument: Instrument.BassGuitar,
    volume: 0.6
  }),
  harmony: new InstrumentConfiguration({
    instrument: Instrument.ElectricPiano,
    volume: 0.8
  }),
  firstBeatArpeggio: new InstrumentConfiguration({
    instrument: Instrument.Harp,
    volume: 0.2
  }),
  melody: new InstrumentConfiguration({
    instrument: Instrument.ElectricPiano,
    octaveShift: 1,
    volume: 0.9
  }),
  melodyOctaves: true
});

/** A very soft preset */
export const Preset3: ProducerPreset = new ProducerPreset({
  bassLine: new InstrumentConfiguration({
    instrument: Instrument.BassGuitar,
    volume: 0.5
  }),
  harmony: new InstrumentConfiguration({
    instrument: Instrument.ElectricPiano,
    volume: 0.6
  }),
  melody: new InstrumentConfiguration({
    instrument: Instrument.SoftPiano,
    octaveShift: 1
  })
});

/** Bass patterns, in tuples [startBeat, duration] */
export const BassPatterns: [number, number][][] = [
  [[0, 4]],
  [[0, 2], [2, 2]],
  [[0, 3], [3, 1]],
  [[0, 3.5], [3.5, 0.5]],
  [[0, 1.5], [1.5, 1.5], [3, 1]]
];
