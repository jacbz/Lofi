import Sortable from 'sortablejs';
import Player, { RepeatMode } from './player';
import Producer from './producer';
import { getRandomOutputParams, OutputParams } from './params';
import { decompress } from './helper';
import generate from './api';

const player = new Player();

// load playlist in URL if possible
const queryString = window.location.search;
if (queryString.length > 0) {
  let outputParams: OutputParams[];
  const compressed = queryString.substring(1);
  try {
    const decompressed = decompress(compressed);
    outputParams = JSON.parse(decompressed);
    // window.history.pushState({}, null, '/');
  } catch (e) {
    console.log('Error parsing', compressed);
  }
  if (outputParams) {
    const playlist = outputParams.map((params) => {
      const producer = new Producer();
      return producer.produce(params);
    });
    player.playlist = playlist;
  }
}

/** Formats seconds into an MM:SS string */
const formatTime = (seconds: number) => {
  if (!seconds || seconds < 0) return '0:00';
  return `${Math.floor(seconds / 60)}:${`0${Math.floor(seconds % 60)}`.slice(-2)}`;
};

// Seekbar
const seekbar = document.getElementById('seekbar') as HTMLInputElement;
seekbar.addEventListener('input', () => {
  timeLabel.textContent = formatTime(seekbar.valueAsNumber);
  formatInputRange(seekbar, '#fc5c8c');
});
let wasPaused = false;
let seekbarDragging = false;
seekbar.addEventListener('mousedown', () => {
  seekbarDragging = true;
  wasPaused = !player.isPlaying;
  if (!wasPaused) {
    player.pause();
  }
});
seekbar.addEventListener('mouseup', () => {
  seekbarDragging = false;
  player.seek(seekbar.valueAsNumber);
  if (!wasPaused) {
    player.play();
  }
});

// Track details and time
const titleLabel = document.getElementById('title');
const timeLabel = document.getElementById('current-time');
const totalTimeLabel = document.getElementById('total-time');
const audio = document.getElementById('audio') as HTMLAudioElement; // dummy audio for Media Session API
const formatInputRange = (input: HTMLInputElement, color: string) => {
  const value = ((input.valueAsNumber - +input.min) / (+input.max - +input.min)) * 100;
  if (!value) {
    input.style.background = 'rgba(0, 0, 0, 0.25)';
  }
  input.style.background = `linear-gradient(to right, ${color} 0%, ${color} ${value}%, rgba(0, 0, 0, 0.25) ${value}%, rgba(0, 0, 0, 0.25) 100%)`;
};
player.updateTrackDisplay = (seconds?: number) => {
  // don't update display while seekbar is being dragged
  if (seekbarDragging) return;

  if (player.currentTrack) {
    vinyl.style.opacity = '1';

    titleLabel.textContent = player.currentTrack.title;
    const totalLength = player.currentTrack.length;
    seekbar.max = `${totalLength}`;
    seekbar.valueAsNumber = +seconds;
    // when current time is within 0.1s of total length, display total length
    timeLabel.textContent = formatTime(seconds);
    totalTimeLabel.textContent = formatTime(totalLength);
  } else {
    vinyl.style.opacity = '0.5';
    titleLabel.textContent = '';
    seekbar.valueAsNumber = 0;
    seekbar.max = '0';
    timeLabel.textContent = '0:00';
    totalTimeLabel.textContent = '0:00';
  }
  formatInputRange(seekbar, '#fc5c8c');
};

// Generate button
const generateButton = document.getElementById('generate-button') as HTMLButtonElement;
const loadingAnimation = document.getElementById('loading-animation');
generateButton.addEventListener('click', async () => {
  generateButton.disabled = true;
  loadingAnimation.style.visibility = 'visible';

  const params = await generate();

  generateButton.disabled = false;
  loadingAnimation.style.visibility = 'hidden';

  const producer = new Producer();
  const track = producer.produce(params);
  player.addToPlaylist(track);
});

// Input field
const inputTextarea = document.getElementById('input') as HTMLTextAreaElement;
inputTextarea.textContent = getRandomOutputParams();

// Add button
const addButton = document.getElementById('add-button');
addButton.addEventListener('click', async () => {
  let params: OutputParams;
  try {
    params = JSON.parse(inputTextarea.value);
    inputTextarea.textContent = getRandomOutputParams();
  } catch (e) {
    window.alert('Could not parse JSON');
    return;
  }
  const producer = new Producer();
  const track = producer.produce(params);
  player.addToPlaylist(track);
});

// On track change
const vinyl = document.getElementById('vinyl');
const vinylColor = document.getElementById('vinyl-color');
const vinylBottomText = document.getElementById('vinyl-bottom-text');
const onTrackChange = () => {
  const trackElements = playlistContainer.querySelectorAll('.track');
  player.playlist.forEach((track, i) => {
    const trackElement = trackElements[i];
    trackElement.classList.toggle('playing', player.currentTrack === track);
  });

  if (player.currentTrack) {
    vinylBottomText.textContent = player.currentTrack.title;
    vinylColor.setAttribute('fill', player.currentTrack.color);
  } else {
    vinylBottomText.textContent = '';
    vinylColor.setAttribute('fill', '#eee');
  }

  /* reset vinyl animation by triggering reflow */
  vinyl.style.animation = 'none';
  const _ = vinyl.offsetHeight;
  vinyl.style.animation = null;
};
player.onTrackChange = onTrackChange;

// Playlist
const playlistContainer = document.getElementById('playlist-tracks');
const updatePlaylistDisplay = () => {
  playlistContainer.innerHTML = '';
  player.playlist.forEach((track, i) => {
    const template = document.getElementById('playlist-track') as HTMLTemplateElement;
    const trackElement = (template.content.cloneNode(true) as HTMLElement).querySelector(
      '.track'
    ) as HTMLDivElement;

    const name = trackElement.querySelector('.track-name');
    name.textContent = track.title;
    const duration = trackElement.querySelector('.track-duration');
    duration.textContent = formatTime(track.length);

    if (track === player.currentTrack) {
      trackElement.classList.add('playing');
    }
    trackElement.addEventListener('click', async (e: MouseEvent) => {
      if ((e.target as HTMLElement).tagName === 'BUTTON') return;
      player.playTrack(i);
      console.log(track.outputParams);
    });

    const deleteButton = trackElement.querySelector('.delete-button');
    deleteButton.addEventListener('click', async () => {
      player.deleteTrack(i);
    });

    playlistContainer.appendChild(trackElement);
  });
};
player.updatePlaylistDisplay = updatePlaylistDisplay;
Sortable.create(playlistContainer, {
  animation: 250,
  ghostClass: 'dragging',
  handle: '.sort-handle',
  onEnd: (event) => {
    const element = player.playlist[event.oldIndex];
    player.playlist.splice(event.oldIndex, 1);
    player.playlist.splice(event.newIndex, 0, element);
    if (player.currentPlayingIndex === event.oldIndex) {
      player.currentPlayingIndex = event.newIndex;
    }
    updatePlaylistDisplay();
  }
});
updatePlaylistDisplay();

// Player controls
const playButton = document.getElementById('play-button');
const playPreviousButton = document.getElementById('play-previous-button');
const playNextButton = document.getElementById('play-next-button');
const repeatButton = document.getElementById('repeat-button');
const shuffleButton = document.getElementById('shuffle-button');
const volumeButton = document.getElementById('volume-button');
const volumeBar = document.getElementById('volume-bar') as HTMLInputElement;
player.getGain = () => volumeBar.valueAsNumber;
const updatePlayingState = () => {
  if (player.isPlaying) {
    playButton.classList.toggle('paused', true);
    vinyl.classList.toggle('paused', false);
    audio.play();
  } else {
    playButton.classList.toggle('paused', false);
    vinyl.classList.toggle('paused', true);
    audio.pause();
  }
};
player.onPlayingStateChange = updatePlayingState;
playButton.addEventListener('click', async () => {
  if (player.isPlaying) {
    player.pause();
  } else {
    player.play();
    if (!player.muted) {
      player.gain.gain.value = volumeBar.valueAsNumber;
    }
  }
});
playPreviousButton.addEventListener('click', async () => {
  player.playPrevious();
});
playNextButton.addEventListener('click', async () => {
  player.playNext();
});
repeatButton.addEventListener('click', async () => {
  switch (player.repeat) {
    case RepeatMode.ALL: {
      player.repeat = RepeatMode.ONE;
      repeatButton.classList.remove('repeat-all');
      repeatButton.classList.add('repeat-one');
      break;
    }
    case RepeatMode.ONE: {
      player.repeat = RepeatMode.NONE;
      repeatButton.classList.remove('repeat-one');
      break;
    }
    default: {
      player.repeat = RepeatMode.ALL;
      repeatButton.classList.add('repeat-all');
      break;
    }
  }
});
shuffleButton.addEventListener('click', async () => {
  player.shuffle = !player.shuffle;
  shuffleButton.classList.toggle('active', player.shuffle);
});
volumeButton.addEventListener('click', async () => {
  player.gain.gain.value = volumeBar.valueAsNumber;
  player.muted = !player.muted;
  volumeButton.classList.toggle('muted', player.muted);
});
volumeBar.addEventListener('input', () => {
  if (player.muted) {
    volumeButton.click();
  }
  if (player.isPlaying) {
    player.gain.gain.value = volumeBar.valueAsNumber;
  }
  formatInputRange(volumeBar, '#fff');
});
formatInputRange(volumeBar, '#fff');

// Export
const exportButton = document.getElementById('export-button');
const exportPanel = document.getElementById('export-panel');
const exportUrlInput = document.getElementById('export-url-input') as HTMLInputElement;
const copyButton = document.getElementById('copy-button');
exportButton.addEventListener('click', async () => {
  exportPanel.style.visibility = 'visible';
  exportPanel.style.opacity = '1';
  const url = player.getExportUrl();
  exportUrlInput.value = url;
  // wait for panel to become visible before we can select the text field
  setTimeout(() => {
    exportUrlInput.select();
  }, 50);
});
exportUrlInput.addEventListener('click', async () => {
  exportUrlInput.select();
});
copyButton.addEventListener('click', async () => {
  document.execCommand('copy');
  exportPanel.style.visibility = 'hidden';
  exportPanel.style.opacity = '0';
});

// Filter panel
// function value(id: string) {
//   return (document.getElementById(id) as HTMLInputElement).valueAsNumber;
// }
// const adjustFilters = () => {
//   player.compressor.threshold.value = value('compressorthreshold');
//   player.compressor.ratio.value = value('compressorratio');
//   player.lowPassFilter.frequency.value = value('lpffrequency');
//   player.lowPassFilter.Q.value = value('lpfq');
//   player.highPassFilter.frequency.value = value('hpffrequency');
//   player.highPassFilter.Q.value = value('hpfq');
//   player.equalizer.low.value = value('eqlow');
//   player.equalizer.mid.value = value('eqmid');
//   player.equalizer.high.value = value('eqhigh');
//   player.reverb.decay = value('reverbdecay');
//   player.reverb.preDelay = value('reverbpredelay');
//   player.reverb.wet.value = value('reverbwet');
//   player.distortion.distortion = value('distortion');
//   player.chebyshev.order = value('chebyshev');
//   // player.bitcrusher.bits.value = value('bitcrusher');

//   const output = new Map();
//   for (const el of document.getElementById('filter').querySelectorAll('input')) {
//     output.set(el.id, el.valueAsNumber);
//   }
//   console.log(output);
// };
// document.getElementById('connecteq').addEventListener('click', () => {
//   player.connectFilter(player.equalizer);
// });
// document.getElementById('connectdistortion').addEventListener('click', () => {
//   player.connectFilter(player.distortion);
// });
// for (const el of document.getElementById('filter').querySelectorAll('input')) {
//   el.addEventListener('input', adjustFilters);
// }

// Media Session API
const actionsAndHandlers = [
  ['play', () => { player.play(); }],
  ['pause', () => { player.pause(); }],
  ['previoustrack', () => { player.playPrevious(); }],
  ['nexttrack', () => { player.playNext(); }],
  ['seekbackward', (details: MediaSessionActionDetails) => { player.seekRelative(-5); }],
  ['seekforward', (details: MediaSessionActionDetails) => { player.seekRelative(5); }],
  ['seekto', (details: MediaSessionActionDetails) => { player.seek(details.seekTime); }],
  ['stop', () => { player.unload(); }]
];
for (const [action, handler] of actionsAndHandlers) {
  try {
    navigator.mediaSession.setActionHandler(action as any, handler as any);
  } catch (error) {
    console.log(`The media session action ${action}, is not supported`);
  }
}
