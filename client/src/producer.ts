import { Track, Loop } from './track';
import { OutputParams } from './params';

abstract class Producer {
  static produceExampleTrack(): Track {
    const track = new Track({
      key: null as any,
      mode: null as any,
      bpm: 70,
      numMeasures: 6,
      drumLoops: [[1, new Loop('0m', '6m')]]
    });
    return track;
  }

  static produce(params: OutputParams): Track {
    return null;
  }
}

export default Producer;
