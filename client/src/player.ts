import * as Tone from 'tone';
import { Track } from './track';
import * as Samples from './samples';

/**
 * Player
 */
class Player {
  /** Current track. Can be undefined */
  currentTrack: Track;

  /** Whether the player is currently playing */
  private _isPlaying: boolean = false;

  get isPlaying() {
    return this._isPlaying;
  }

  set isPlaying(isPlaying: boolean) {
    this._isPlaying = isPlaying;
    this.onPlayingStateChange(isPlaying);
    if (this.gain) {
      this.gain.gain.value = +isPlaying;
    }
  }

  /** Function to update track information in the UI */
  updateTrackDisplay: (seconds: number) => void;

  /** Function to call when isPlaying changes */
  onPlayingStateChange: (isPlaying: boolean) => void;

  samplePlayers: Map<string, Tone.Player[]>;

  instrumentSamplers: Map<string, Tone.Sampler>;

  /** Filters */

  compressor: Tone.Compressor;

  lowPassFilter: Tone.Filter;

  highPassFilter: Tone.Filter;

  equalizer: Tone.EQ3;

  distortion: Tone.Distortion;

  reverb: Tone.Reverb;

  chebyshev: Tone.Chebyshev;

  bitcrusher: Tone.BitCrusher;

  gain: Tone.Gain;

  filters: Tone.ToneAudioNode[];

  initFilters() {
    this.compressor = new Tone.Compressor(0, 1);
    this.lowPassFilter = new Tone.Filter({
      type: 'lowpass',
      frequency: 5000
    });
    this.highPassFilter = new Tone.Filter({
      type: 'highpass',
      frequency: 0
    });
    this.equalizer = new Tone.EQ3(0, 0, 0);
    this.distortion = new Tone.Distortion(0);
    this.reverb = new Tone.Reverb({
      decay: 0.001,
      wet: 0,
      preDelay: 0
    });
    this.chebyshev = new Tone.Chebyshev(1);
    // this.bitcrusher = new Tone.BitCrusher(16);
    this.gain = new Tone.Gain();

    this.filters = [
      this.compressor,
      this.lowPassFilter,
      this.highPassFilter,
      this.reverb,
      // this.bitcrusher,
      // this.equalizer,
      this.chebyshev,
      // this.distortion,
      this.gain
    ];
  }

  connectFilter(filter: Tone.ToneAudioNode) {
    this.filters.splice(this.filters.indexOf(this.gain), 0, filter);
    for (const player of this.instrumentSamplers.values()) {
      player.disconnect();
      player.chain(...this.filters, Tone.Destination);
    }
    for (const player of this.samplePlayers.values()) {
      for (const player2 of player.values()) {
        if (!player2) return;
        player2.disconnect();
        player2.chain(...this.filters, Tone.Destination);
      }
    }
  }

  async play() {
    if (!this.currentTrack) {
      return;
    }
    this.isPlaying = true;

    Tone.Transport.cancel();
    Tone.Transport.bpm.value = this.currentTrack.bpm;

    this.samplePlayers = new Map();
    this.instrumentSamplers = new Map();

    this.initFilters();

    // load samples
    for (const [sampleGroupName, sampleIndex] of this.currentTrack.samples) {
      const sampleGroup = Samples.LOOPS.get(sampleGroupName);
      const player = new Tone.Player({
        url: sampleGroup.getSampleUrl(sampleIndex),
        volume: sampleGroup.volume,
        loop: true,
        fadeIn: '4n',
        fadeOut: '4n',
        // TODO: don't change pitch
        playbackRate: sampleGroup.bpm ? this.currentTrack.bpm / sampleGroup.bpm[sampleIndex] : 1.0
      })
        .chain(...this.filters, Tone.Destination)
        .sync();

      if (!this.samplePlayers.has(sampleGroupName)) {
        this.samplePlayers.set(sampleGroupName, Array(sampleGroup.size));
      }
      this.samplePlayers.get(sampleGroupName)[sampleIndex] = player;
    }

    // load instruments
    for (const instrumentName of this.currentTrack.instruments) {
      const instrument = Samples.SAMPLE_INSTRUMENTS.get(instrumentName);
      const sampler = new Tone.Sampler({
        urls: instrument.map,
        baseUrl: `${Samples.SAMPLES_BASE_URL}/instruments/${instrument.name}/`,
        volume: instrument.volume
      })
        .chain(...this.filters, Tone.Destination)
        .sync();
      this.instrumentSamplers.set(instrumentName, sampler);
    }

    // wait until all samples are loaded
    await Tone.loaded();
    await this.reverb.generate();

    for (const sampleLoop of this.currentTrack.sampleLoops) {
      const samplePlayer = this.samplePlayers.get(sampleLoop.sampleGroupName)[
        sampleLoop.sampleIndex
      ];
      samplePlayer.start(sampleLoop.startTime);
      samplePlayer.stop(sampleLoop.stopTime);
    }

    for (const noteTiming of this.currentTrack.instrumentNotes) {
      const instrumentSampler = this.instrumentSamplers.get(noteTiming.instrument);
      instrumentSampler.triggerAttackRelease(
        noteTiming.pitch,
        noteTiming.duration,
        noteTiming.time
      );
    }

    Tone.Transport.scheduleRepeat((time) => {
      const seconds = Tone.Transport.getSecondsAtTime(time);
      this.updateTrackDisplay(seconds);

      if (this.currentTrack.length - seconds < 0) {
        Tone.Transport.stop();
        this.isPlaying = false;
      }
    }, 0.1);

    Tone.Transport.start();
  }

  seek(seconds: number) {
    if (this.currentTrack) {
      Tone.Transport.seconds = seconds;
      this.updateTrackDisplay(seconds);
    }
  }

  continue() {
    if (this.currentTrack) {
      this.isPlaying = true;
      Tone.Transport.start();
      this.seek(Tone.Transport.seconds);
    }
  }

  pause() {
    this.isPlaying = false;
    Tone.Transport.pause();
  }
}

export default Player;
