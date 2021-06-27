import * as Tone from 'tone';
import { OutputParams } from './params';
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

// Time
const timeLabel = document.getElementById('time');
const totalTimeLabel = document.getElementById('total-time');
player.updateDisplayTime = (seconds: number) => {
  const totalLength = player.currentTrack.length;
  seekbar.max = `${totalLength}`;

  seekbar.valueAsNumber = seconds;
  // when current time is within 0.1s of total length, display total length
  timeLabel.textContent = formatTime(seconds);
  totalTimeLabel.textContent = formatTime(totalLength);
};

// Input field
const inputTextarea = document.getElementById('input') as HTMLTextAreaElement;

// Play button
const playButton = document.getElementById('play-button');
const updatePlayingState = (isPlaying: boolean) => {
  if (isPlaying) {
    playButton.textContent = 'Pause';
  } else {
    playButton.textContent = 'Play';
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
    const track = Producer.produce(params);
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
