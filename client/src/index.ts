import * as Tone from 'tone';
import { DEFAULT_INPUT_PARAMS, OutputParams } from './params';
import Player from './player';
import Producer from './producer';

const player = new Player();

/** Formats seconds into an MM:SS string */
function formatTime(seconds: number) {
  const format = (val: number) => `0${Math.floor(val)}`.slice(-2);
  const minutes = (seconds % 3600) / 60;
  if (minutes < 0) return '00:00';
  return [minutes, seconds % 60].map(format).join(':');
}

// Seekbar
const seekbar = document.getElementById('seekbar') as HTMLInputElement;
seekbar.addEventListener('input', () => {
  player.seek(seekbar.valueAsNumber);
});

// Track details and time
const titleLabel = document.getElementById('title');
const timeLabel = document.getElementById('time');
const totalTimeLabel = document.getElementById('total-time');
player.updateTrackDisplay = (seconds: number) => {
  titleLabel.textContent = player.currentTrack.title;
  const totalLength = player.currentTrack.length;
  seekbar.max = `${totalLength}`;

  seekbar.valueAsNumber = seconds;
  // when current time is within 0.1s of total length, display total length
  timeLabel.textContent = formatTime(seconds);
  totalTimeLabel.textContent = formatTime(totalLength);
};

// Input field
const inputTextarea = document.getElementById('input') as HTMLTextAreaElement;
inputTextarea.textContent = JSON.stringify(
  DEFAULT_INPUT_PARAMS,
  (k, v) => (v instanceof Array ? JSON.stringify(v) : v),
  2
);

// Play button
const playButton = document.getElementById('play-button');
const updatePlayingState = (isPlaying: boolean) => {
  if (isPlaying) {
    playButton.classList.toggle('paused', true);
  } else {
    playButton.classList.toggle('paused', false);
  }
};
player.onPlayingStateChange = updatePlayingState;
playButton.addEventListener('click', async () => {
  await Tone.start();
  if (!player.currentTrack) {
    let params: OutputParams;
    try {
      params = JSON.parse(inputTextarea.value);
    } catch (e) {
      window.alert('Could not parse JSON');
      return;
    }
    const producer = new Producer();
    const track = producer.produce(params);
    player.currentTrack = track;
    await player.play();
    return;
  }
  if (player.isPlaying) {
    player.pause();
  } else {
    player.continue();
  }
});

// filter panel
function value(id: string) {
  return (document.getElementById(id) as HTMLInputElement).valueAsNumber;
}
const adjustFilters = () => {
  player.compressor.threshold.value = value('compressorthreshold');
  player.compressor.ratio.value = value('compressorratio');
  player.lowPassFilter.frequency.value = value('lpffrequency');
  player.lowPassFilter.Q.value = value('lpfq');
  player.highPassFilter.frequency.value = value('hpffrequency');
  player.highPassFilter.Q.value = value('hpfq');
  player.equalizer.low.value = value('eqlow');
  player.equalizer.mid.value = value('eqmid');
  player.equalizer.high.value = value('eqhigh');
  player.reverb.decay = value('reverbdecay');
  player.reverb.preDelay = value('reverbpredelay');
  player.reverb.wet.value = value('reverbwet');
  player.distortion.distortion = value('distortion');
  player.chebyshev.order = value('chebyshev');
  // player.bitcrusher.bits.value = value('bitcrusher');
  player.gain.gain.value = value('gain');

  const output = new Map();
  for (const el of document.getElementById('filter').querySelectorAll('input')) {
    output.set(el.id, el.valueAsNumber);
  }
  console.log(output);
};

document.getElementById('connecteq').addEventListener('click', () => {
  player.connectFilter(player.equalizer);
});
document.getElementById('connectdistortion').addEventListener('click', () => {
  player.connectFilter(player.distortion);
});

for (const el of document.getElementById('filter').querySelectorAll('input')) {
  el.addEventListener('input', adjustFilters);
}
