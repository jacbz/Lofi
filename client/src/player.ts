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

    const drumPlayers: Map<number, Tone.Player> = new Map();

    // load samples
    for (const [drumLoopId, loop] of this.currentTrack.drumLoops) {
      const drumLoop = Samples.DRUM_LOOPS.get(drumLoopId);
      const player = new Tone.Player({
        url: drumLoop.url,
        volume: drumLoop.volume,
        loop: true
      }).toDestination();
      drumPlayers.set(drumLoopId, player);
    }

    // wait until all samples are loaded
    await Tone.loaded();

    for (const [drumLoopId, loop] of this.currentTrack.drumLoops) {
      const drumPlayer = drumPlayers.get(drumLoopId);
      drumPlayer.sync().start(loop.startTime);
      drumPlayer.sync().stop(loop.stopTime);
    }

    // const synthA = new Tone.AMSynth().toDestination();
    // const loopA = new Tone.Loop((time) => {
    //   synthA.triggerAttackRelease('C3', '8n', time);
    // }, '1m');
    // loopA.start(0);

    Tone.Transport.scheduleRepeat((time) => {
      const seconds = Tone.Transport.getSecondsAtTime(time);
      this.updateDisplayTime(seconds);

      if (this.currentTrack.length - seconds < 0.1) {
        Tone.Transport.stop();
        this.isPlaying = false;
      }
    }, 0.1);

    Tone.Transport.start();
  }

  seek(seconds: number) {
    if (this.currentTrack) {
      Tone.Transport.seconds = seconds;
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
