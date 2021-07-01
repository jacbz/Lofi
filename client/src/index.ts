import Player from './player';
import Producer from './producer';
import { getRandomInputParams, OutputParams } from './params';

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
inputTextarea.textContent = getRandomInputParams();

// Add button
const addButton = document.getElementById('add-button');
addButton.addEventListener('click', async () => {
  let params: OutputParams;
  try {
    params = JSON.parse(inputTextarea.value);
    console.log(params);
    inputTextarea.textContent = getRandomInputParams();
  } catch (e) {
    window.alert('Could not parse JSON');
    return;
  }
  const producer = new Producer();
  const track = producer.produce(params);
  await player.addToPlaylist(track);
});

// Playlist
const playlistContainer = document.getElementById('playlist');
const updatePlaylistDisplay = () => {
  playlistContainer.innerHTML = '';
  for (const track of player.playlist) {
    const template = document.getElementById('playlist-entry') as HTMLTemplateElement;
    const trackElement = (template.content.cloneNode(true) as HTMLElement).querySelector('.track') as HTMLDivElement;

    const name = trackElement.querySelector('.track-name');
    name.textContent = track.title;
    const duration = trackElement.querySelector('.track-duration');
    duration.textContent = formatTime(track.length);

    if (track === player.currentTrack) {
      trackElement.classList.add('playing');
    }
    trackElement.addEventListener('click', async () => {
      player.playTrack(track);
    });

    playlistContainer.appendChild(trackElement);
  }
};
player.updatePlaylistDisplay = updatePlaylistDisplay;

// Play button
const playButton = document.getElementById('play-button');
const vinyl = document.getElementById('vinyl');
const updatePlayingState = (isPlaying: boolean) => {
  if (isPlaying) {
    playButton.classList.toggle('paused', true);
    vinyl.classList.toggle('paused', false);
  } else {
    playButton.classList.toggle('paused', false);
    vinyl.classList.toggle('paused', true);
  }
};
player.onPlayingStateChange = updatePlayingState;
playButton.addEventListener('click', async () => {
  if (player.isPlaying) {
    player.pause();
  } else {
    player.continue();
  }
});

// Filter panel
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
