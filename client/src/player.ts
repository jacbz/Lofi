/* eslint-disable class-methods-use-this */
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
  }

  /** Function to update the time in the UI */
  updateDisplayTime: (seconds: number) => void;

  /** Function to call when isPlaying changes */
  onPlayingStateChange: (isPlaying: boolean) => void;

  async play() {
    if (!this.currentTrack) {
      return;
    }
    this.isPlaying = true;

    Tone.Transport.cancel();
    Tone.Transport.bpm.value = this.currentTrack.bpm;

    const samplePlayers: Map<string, Tone.Player[]> = new Map();
    const instrumentSamplers: Map<string, Tone.Sampler> = new Map();

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
        playbackRate: sampleGroup.bpm ? this.currentTrack.bpm / sampleGroup.bpm : 1.0
      }).toDestination().sync();
      if (!samplePlayers.has(sampleGroupName)) {
        samplePlayers.set(sampleGroupName, Array(sampleGroup.size));
      }
      samplePlayers.get(sampleGroupName)[sampleIndex] = player;
    }

    // load instruments
    for (const instrumentName of this.currentTrack.instruments) {
      const instrument = Samples.SAMPLE_INSTRUMENTS.get(instrumentName);
      const sampler = new Tone.Sampler({
        urls: instrument.map,
        baseUrl: `${Samples.SAMPLES_BASE_URL}/instruments/${instrument.name}/`,
        volume: instrument.volume
      }).toDestination().sync();
      instrumentSamplers.set(instrumentName, sampler);
    }

    // wait until all samples are loaded
    await Tone.loaded();

    for (const sampleLoop of this.currentTrack.sampleLoops) {
      const samplePlayer = samplePlayers.get(sampleLoop.sampleGroupName)[sampleLoop.sampleIndex];
      samplePlayer.start(sampleLoop.startTime);
      samplePlayer.stop(sampleLoop.stopTime);
    }

    for (const noteTiming of this.currentTrack.instrumentNotes) {
      const instrumentSampler = instrumentSamplers.get(noteTiming.instrument);
      instrumentSampler.triggerAttackRelease(
        noteTiming.pitch,
        noteTiming.duration,
        noteTiming.time
      );
    }

    Tone.Transport.scheduleRepeat((time) => {
      const seconds = Tone.Transport.getSecondsAtTime(time);
      this.updateDisplayTime(seconds);

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
      this.updateDisplayTime(seconds);
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
