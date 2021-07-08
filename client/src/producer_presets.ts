import { Instrument } from './instruments';

export class InstrumentConfiguration {
  instrument: Instrument;

  volume = 1;

  octaveShift = 0;

  public constructor(init?: Partial<InstrumentConfiguration>) {
    Object.assign(this, init);
  }
}

export class ProducerPreset {
  /** Instrument that plays the root note at the first beat beginning of each chord */
  bassLine: InstrumentConfiguration;

  /** Instrument that plays the entire chord at the first beat of each chord */
  harmony: InstrumentConfiguration;

  /** Instrument that plays an arpeggio at the first beat */
  firstBeatArpeggio: InstrumentConfiguration;

  firstBeatArpeggioPattern = [1, 5, 8, 9, 10];

  /** Instrument that plays the diminished chord arpeggio at the second beat */
  secondBeatArpeggio: InstrumentConfiguration;

  /** Instrument that plays the melody */
  melody: InstrumentConfiguration;

  /** Play the melody in octaves */
  melodyOctaves = false;

  public constructor(init?: Partial<ProducerPreset>) {
    Object.assign(this, init);
  }
}

export const selectPreset = (valence: number, energy: number): ProducerPreset => {
  if (energy <= 0.5) {
    return Preset2;
  }
  return Preset1;
};

export const Preset1: ProducerPreset = new ProducerPreset({
  bassLine: new InstrumentConfiguration({
    instrument: Instrument.BassGuitar,
    octaveShift: -1,
    volume: 0.6
  }),
  harmony: new InstrumentConfiguration({
    instrument: Instrument.ElectricPiano,
    octaveShift: 1
  }),
  firstBeatArpeggio: new InstrumentConfiguration({
    instrument: Instrument.Piano,
    octaveShift: -1,
    volume: 0.3
  }),
  // secondBeatArpeggio: new InstrumentConfiguration({
  //   instrument: Instrument.Piano,
  //   volume: 0.3
  // }),
  melody: new InstrumentConfiguration({
    instrument: Instrument.AcousticGuitar,
    octaveShift: -1,
    volume: 0.8
  })
});

/** A softer preset, with piano melodies */
export const Preset2: ProducerPreset = new ProducerPreset({
  bassLine: new InstrumentConfiguration({
    instrument: Instrument.BassGuitar,
    octaveShift: -1,
    volume: 0.6
  }),
  harmony: new InstrumentConfiguration({
    instrument: Instrument.ElectricPiano,
    octaveShift: 1
  }),
  firstBeatArpeggio: new InstrumentConfiguration({
    instrument: Instrument.AcousticGuitar,
    octaveShift: -1,
    volume: 0.3
  }),
  // secondBeatArpeggio: new InstrumentConfiguration({
  //   instrument: Instrument.Piano,
  //   volume: 0.3
  // }),
  melody: new InstrumentConfiguration({
    instrument: Instrument.Piano,
    octaveShift: 1,
    volume: 0.8
  }),
  melodyOctaves: true
});
