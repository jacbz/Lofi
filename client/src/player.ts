import * as Tone from 'tone';
import { getInstrumentFilters, getInstrumentSampler, Instrument } from './instruments';
import * as Samples from './samples';
import { Track } from './track';
import { compress } from './helper';

/**
 * The Player plays a Track through Tone.js.
 */
class Player {
  playlist: Track[] = [];

  currentPlayingIndex: number;

  /** Current track. Can be undefined */
  get currentTrack() {
    if (this.currentPlayingIndex !== undefined) {
      return this.playlist[this.currentPlayingIndex];
    }
    return undefined;
  }

  /** Whether the player is currently playing */
  private _isPlaying: boolean = false;

  get isPlaying() {
    return this._isPlaying;
  }

  set isPlaying(isPlaying: boolean) {
    this._isPlaying = isPlaying;
    this.onPlayingStateChange();
    if (this.gain) {
      this.gain.gain.value = isPlaying ? this.getGain() : 0;
    }
  }

  repeat: RepeatMode = RepeatMode.NONE;

  shuffle = false;

  /** Playing queue, used when shuffling */
  shuffleQueue: number[] = [];

  private _muted = false;

  get muted() {
    return this._muted;
  }

  set muted(muted: boolean) {
    this._muted = muted;
    this.gain.gain.value = muted ? 0 : this.getGain();
  }

  /** Function to get the gain from the UI */
  getGain: () => number;

  /** Function to update the playlist in the UI */
  updatePlaylistDisplay: () => void;

  /** Function to update track information in the UI */
  updateTrackDisplay: (seconds?: number, spectrum?: Float32Array) => void;

  /** Function to call when the track changes */
  onTrackChange: () => void;

  /** Function to call when isPlaying changes */
  onPlayingStateChange: () => void;

  samplePlayers: Map<string, Tone.Player[]>;

  instrumentSamplers: Map<Instrument, Tone.Sampler>;

  gain: Tone.Gain;

  /** Adds a given track to the playlist */
  addToPlaylist(track: Track) {
    this.playlist.push(track);
    this.updatePlaylistDisplay();
    if (!this.isPlaying) {
      this.playTrack(this.playlist.length - 1);
    }
    this.fillShuffleQueue();
  }

  /** Plays a specific track in the playlist */
  playTrack(playlistIndex: number) {
    this.currentPlayingIndex = playlistIndex;
    this.onTrackChange();
    this.seek(0);
    this.stop();
    this.load();
  }

  /** Sets up Tone.Transport for the current track and starts playback */
  async load() {
    if (!this.currentTrack) {
      return;
    }
    this.gain = new Tone.Gain();
    this.isPlaying = true;
    this.setAudioWebApiMetadata();

    // wait 500ms before trying to play the track
    // this is needed due to Tone.js scheduling conflicts if the user rapidly changes the track
    const trackToPlayIndex = this.currentPlayingIndex;
    await new Promise((resolve) => setTimeout(resolve, 500));
    if (trackToPlayIndex !== this.currentPlayingIndex || !this.isPlaying) {
      return;
    }

    await Tone.start();
    Tone.Transport.bpm.value = this.currentTrack.bpm;

    this.samplePlayers = new Map();
    this.instrumentSamplers = new Map();

    // this.initDefaultFilters();
    // load samples
    for (const [sampleGroupName, sampleIndex] of this.currentTrack.samples) {
      const sampleGroup = Samples.SAMPLEGROUPS.get(sampleGroupName);
      const filters = sampleGroup.getFilters();
      // if the sample group specifies a specific key, shift to that key
      // if (sampleGroup.keys && sampleGroup.keys[sampleIndex] !== this.currentTrack.keyNum) {
      //   const shift = pitchShiftDistance(
      //     keyNumberToString(sampleGroup.keys[sampleIndex] - 1), this.currentTrack.key
      //   );
      //   filters.push(new PitchShift(shift));
      // }
      const player = new Tone.Player({
        url: sampleGroup.getSampleUrl(sampleIndex),
        volume: sampleGroup.volume,
        loop: sampleGroup.isLoop
        // fadeIn: '4n',
        // fadeOut: sampleGroup.isLoop ? 0 : '4n'
      })
        .chain(...filters, this.gain, Tone.Destination)
        .sync();

      if (!this.samplePlayers.has(sampleGroupName)) {
        this.samplePlayers.set(sampleGroupName, Array(sampleGroup.size));
      }
      this.samplePlayers.get(sampleGroupName)[sampleIndex] = player;
    }

    // load instruments
    for (const instrument of this.currentTrack.instruments) {
      const sampler = getInstrumentSampler(instrument)
        .chain(...getInstrumentFilters(instrument), this.gain, Tone.Destination)
        .sync();
      this.instrumentSamplers.set(instrument, sampler);
    }

    // set swing
    Tone.Transport.swing = this.currentTrack.swing ? 2 / 3 : 0;

    // wait until all samples are loaded
    await Tone.loaded();
    // await this.reverb.generate();

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
        noteTiming.time,
        noteTiming.velocity !== undefined ? noteTiming.velocity : 1
      );
    }

    const analyzer = new Tone.Analyser('fft', 16);
    this.gain.connect(analyzer);

    Tone.Transport.scheduleRepeat((time) => {
      const seconds = Tone.Transport.getSecondsAtTime(time);
      const spectrum = analyzer.getValue() as Float32Array;
      this.updateTrackDisplay(seconds, spectrum);
      this.updateAudioWebApiPosition(seconds);

      if (this.currentTrack.length - seconds < 0) {
        this.playNext();
      }
    }, 0.1);

    this.play();
  }

  /** Starts playback on the current track; the track must have been loaded */
  play() {
    if (this.currentTrack) {
      this.isPlaying = true;
      Tone.Transport.start();
      this.seek(Tone.Transport.seconds);
    } else if (this.playlist.length > 0) {
      this.playTrack(0);
    }
  }

  /** Seeks to a specific position in the current track */
  seek(seconds: number) {
    if (!this.currentTrack) return;
    this.instrumentSamplers?.forEach((s) => s.releaseAll());
    Tone.Transport.seconds = seconds;
    this.updateTrackDisplay(seconds);
  }

  /** Seeks to a specific position in the current track, relative to the current position */
  seekRelative(seconds: number) {
    if (!this.currentTrack) return;
    const position = Math.max(0, Tone.Transport.seconds + seconds);
    if (position > this.currentTrack.length) {
      this.stop();
    }
    this.seek(position);
  }

  /** Pauses the current track */
  pause() {
    this.isPlaying = false;
    Tone.Transport.pause();
  }

  /** Stops the current track, and disposes Tone.js objects */
  stop() {
    this.isPlaying = false;
    this.gain?.disconnect();
    Tone.Transport.cancel();
    Tone.Transport.stop();
    this.instrumentSamplers?.forEach((s) => s.dispose());
    this.samplePlayers?.forEach((s) => s.forEach((t) => t.dispose()));
  }

  /** Stops playback and unloads the current track in the UI */
  unload() {
    this.stop();
    this.currentPlayingIndex = undefined;
    this.updateTrackDisplay();
    this.updatePlaylistDisplay();
    navigator.mediaSession.metadata = null;
  }

  /** Plays the previous track */
  playPrevious() {
    let nextTrackIndex = null;
    if (this.currentPlayingIndex > 0) {
      nextTrackIndex = this.currentPlayingIndex - 1;
    } else if (this.currentPlayingIndex === 0) {
      if (this.repeat === RepeatMode.ALL) {
        nextTrackIndex = this.playlist.length - 1;
      } else {
        this.seek(0);
      }
    }

    if (nextTrackIndex !== null) {
      this.playTrack(nextTrackIndex);
    }
  }

  /** Plays the next track */
  playNext() {
    if (this.repeat === RepeatMode.ONE) {
      this.seek(0);
      return;
    }

    let nextTrackIndex = null;
    if (this.shuffle) {
      if (this.shuffleQueue.length === 0) this.fillShuffleQueue();
      nextTrackIndex = this.shuffleQueue.shift();
    } else if (this.currentPlayingIndex < this.playlist.length - 1) {
      nextTrackIndex = this.currentPlayingIndex + 1;
    } else if (
      this.currentPlayingIndex === this.playlist.length - 1 &&
      this.repeat === RepeatMode.ALL
    ) {
      nextTrackIndex = 0;
    }

    if (nextTrackIndex !== null) {
      this.playTrack(nextTrackIndex);
    } else {
      this.unload();
    }
  }

  /** Generates a 'shuffle queue' */
  fillShuffleQueue() {
    this.shuffleQueue = [...Array(this.playlist.length).keys()];

    // shuffle
    for (let i = this.shuffleQueue.length - 1; i > 0; i -= 1) {
      const j = Math.floor(Math.random() * (i + 1));
      [this.shuffleQueue[i], this.shuffleQueue[j]] = [this.shuffleQueue[j], this.shuffleQueue[i]];
    }
  }

  /** Deletes a track from the playlist */
  deleteTrack(index: number) {
    this.playlist.splice(index, 1);
    if (index === this.currentPlayingIndex) {
      this.unload();
    } else if (index < this.currentPlayingIndex) {
      this.currentPlayingIndex -= 1;
    }
    this.updatePlaylistDisplay();
  }

  /** Generate a URL that points to the current playlist */
  getExportUrl() {
    const json = JSON.stringify(this.playlist.map((t) => t.outputParams));
    const compressed = compress(json);
    return `${window.location.origin}${window.location.pathname}?${compressed}`.replace(
      'home.in.tum.de/~zhangja/lofi',
      'lofi.jacobzhang.de'
    );
  }

  /** Set up Media Session API metadata */
  setAudioWebApiMetadata() {
    if (!('mediaSession' in navigator) || !this.currentTrack) return;
    navigator.mediaSession.metadata = new MediaMetadata({
      title: this.currentTrack.title,
      artist: 'Lofi generator',
      artwork: [{ src: './background.jpg', type: 'image/jpg' }]
    });
    this.updateAudioWebApiPosition(0);
  }

  /** Set up Media Session API current position */
  updateAudioWebApiPosition(seconds: number) {
    if (!('mediaSession' in navigator) || !this.currentTrack) return;
    navigator.mediaSession.setPositionState({
      duration: this.currentTrack.length,
      position: Math.max(0, Math.min(this.currentTrack.length, seconds))
    });
  }
}

export enum RepeatMode {
  NONE,
  ALL,
  ONE
}

export default Player;
