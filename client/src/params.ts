class InputParams {
  text: string;
}

class OutputParams {
  chordProgression: ChordDTO[]
}

class ChordDTO {
  /** Scale degree */
  sd: number
}

export { InputParams, OutputParams };
