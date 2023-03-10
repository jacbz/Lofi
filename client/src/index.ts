import Sortable from 'sortablejs';
import Player, { RepeatMode } from './player';
import Producer from './producer';
import { DEFAULT_OUTPUTPARAMS, HIDDEN_SIZE, OutputParams } from './params';
import { decompress, randn } from './helper';
import { decode } from './api';

const player = new Player();

// check if local storage is available
let localStorageAvailable = false;
try {
  const x = '__storage_test__';
  window.localStorage.setItem(x, x);
  window.localStorage.removeItem(x);
  localStorageAvailable = true;
} catch (e) {
  console.log('Local storage is unavailable');
}

// try to load playlist from local storage
let playlistToLoad: OutputParams[] = [];
if (localStorageAvailable) {
  const localStoragePlaylist = localStorage.getItem('playlist');
  if (localStoragePlaylist) {
    try {
      playlistToLoad = JSON.parse(localStoragePlaylist);
    } catch (e) {
      console.log('Error parsing', localStoragePlaylist);
    }
  }
}
const updateLocalStorage = () => {
  if (localStorageAvailable) {
    localStorage.setItem('playlist', JSON.stringify(player.playlist.map((t) => t.outputParams)));
  }
};
player.updateLocalStorage = updateLocalStorage;

// load playlist in URL if possible
const queryString = window.location.search;
if (queryString.length > 0) {
  const compressedPlaylist = queryString === '?default' ? DEFAULT_OUTPUTPARAMS : queryString.substring(1);
  try {
    const decompressed = decompress(compressedPlaylist);
    const outputParams: OutputParams[] = JSON.parse(decompressed);
    playlistToLoad = [
      ...playlistToLoad.filter((p) => outputParams.every((p2) => p2.title !== p.title)),
      ...outputParams
    ];
    window.history.pushState({}, null, window.location.href.split('?')[0]);
  } catch (e) {
    console.log('Error parsing', compressedPlaylist);
  }
}

if (playlistToLoad.length > 0) {
  const playlist = playlistToLoad.map((params) => {
    const producer = new Producer();
    return producer.produce(params);
  });
  player.playlist = playlist;
  updateLocalStorage();
}

// Sliders
const slidersEl = document.getElementById('sliders');
const sliders: HTMLInputElement[] = [];
for (let i = 0; i < HIDDEN_SIZE; i += 1) {
  const slider = document.createElement('input') as HTMLInputElement;
  slider.type = 'range';
  slider.min = '-4';
  slider.max = '4';
  slider.step = '0.01';
  slider.valueAsNumber = randn();
  slidersEl.appendChild(slider);
  sliders.push(slider);
}

// Help button
const helpButton = document.getElementById('help');
const introText = document.getElementById('intro-text');
helpButton.addEventListener('click', () => {
  if (introText.style.maxHeight) {
    introText.style.maxHeight = null;
  } else {
    introText.style.maxHeight = '200px';
  }
});

// Refresh Button
const refreshButton = document.getElementById('refresh-button');
export function refreshLatentSpace() {
  sliders.forEach((s) => {
    s.valueAsNumber = randn();
  });
}
refreshButton.addEventListener('click', refreshLatentSpace);

// Generate button
const generateButton = document.getElementById('generate-button') as HTMLButtonElement;
const loadingAnimation = document.getElementById('loading-animation');
export async function generateNewTrack() {
  generateButton.disabled = true;
  loadingAnimation.style.display = null;

  const numberArray = sliders.map((n) => n.valueAsNumber);

  let params;
  try {
    params = await decode(numberArray);
  } catch (err) {
    generateButton.textContent = 'Error!';
    return;
  }
  const producer = new Producer();
  const track = producer.produce(params);
  player.addToPlaylist(track);
  // scroll to end of playlist
  playlistContainer.scrollTop = playlistContainer.scrollHeight;

  generateButton.disabled = false;
  loadingAnimation.style.display = 'none';
}

generateButton.addEventListener('click', generateNewTrack);

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
['mousedown', 'touchstart'].forEach((e) => seekbar.addEventListener(e, () => {
  seekbarDragging = true;
  wasPaused = !player.isPlaying;
  if (!wasPaused) {
    player.pause();
  }
}));
['mouseup', 'touchend'].forEach((e) => seekbar.addEventListener(e, () => {
  seekbarDragging = false;
  player.seek(seekbar.valueAsNumber);
  if (!wasPaused) {
    player.play();
  }
}));

// Visualizer
const visualizer = document.getElementById('visualizer');
const spectrumBars: HTMLDivElement[] = [];
for (let i = 0; i < 22; i += 1) {
  const spectrumBar = document.createElement('div');
  spectrumBar.classList.add('spectrum-bar');
  visualizer.appendChild(spectrumBar);
  spectrumBars.push(spectrumBar);
}
const minDecibels = -100;
const maxDecibels = -10;
const updateVisualization = (spectrum: Float32Array) => {
  spectrumBars.forEach((bar: HTMLDivElement, i) => {
    if (spectrum) {
      const val = Math.min(maxDecibels, Math.max(minDecibels, spectrum[i]));
      const scaled = (100 / (maxDecibels - minDecibels)) * (val - minDecibels);
      bar.style.height = `${scaled}%`;
    } else {
      bar.style.height = '0%';
    }
  });
};

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
player.updateTrackDisplay = (seconds?: number, spectrum?: Float32Array) => {
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
    vinyl.style.transform = `rotate(${seconds * 8}deg)`;
  } else {
    vinyl.style.opacity = '0.5';
    titleLabel.textContent = '';
    seekbar.valueAsNumber = 0;
    seekbar.max = '0';
    timeLabel.textContent = '0:00';
    totalTimeLabel.textContent = '0:00';
  }
  formatInputRange(seekbar, '#fc5c8c');
  updateVisualization(spectrum);
};

// On track change
const vinyl = document.getElementById('vinyl');
const vinylColor = document.getElementById('vinyl-color');
const vinylBottomText1 = document.getElementById('vinyl-bottom-text1');
const vinylBottomText2 = document.getElementById('vinyl-bottom-text2');
const loadPlaylistButton = document.getElementById('load-playlist-button');
const playlistContainer = document.getElementById('playlist-tracks');
const updateTrackClasses = () => {
  player.playlist.forEach((track, i) => {
    const trackElements = playlistContainer.querySelectorAll('.track');
    const trackElement = trackElements[i];
    trackElement.classList.toggle('playing', player.currentTrack === track);
    trackElement.classList.toggle('loading', player.currentTrack === track && player.isLoading);
  });
};
const onTrackChange = () => {
  updateTrackClasses();

  if (player.currentTrack) {
    vinylBottomText1.textContent = `${player.currentTrack.key} ${player.currentTrack.mode}`;
    vinylBottomText2.textContent = player.currentTrack.title.substring(0, 10);
    vinylColor.setAttribute('fill', player.currentTrack.color);
  } else {
    vinylBottomText1.textContent = '';
    vinylBottomText2.textContent = '';
    vinylColor.setAttribute('fill', '#eee');
  }
};
player.onTrackChange = onTrackChange;

loadPlaylistButton.addEventListener('click', () => {
  window.location.href = '?default';
});

// Playlist
const updatePlaylistDisplay = () => {
  loadPlaylistButton.style.display = player.playlist.length > 0 ? 'none' : null;
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
  delay: 400,
  delayOnTouchOnly: true,
  ghostClass: 'dragging',
  onEnd: (event) => {
    const element = player.playlist[event.oldIndex];
    player.playlist.splice(event.oldIndex, 1);
    player.playlist.splice(event.newIndex, 0, element);
    if (player.currentPlayingIndex === event.oldIndex) {
      player.currentPlayingIndex = event.newIndex;
    }
    updatePlaylistDisplay();
    updateLocalStorage();
  }
});
updatePlaylistDisplay();

// Player controls
const playButton = document.getElementById('play-button');
const playPreviousButton = document.getElementById('play-previous-button');
const playNextButton = document.getElementById('play-next-button');
const repeatButton = document.getElementById('repeat-button') as HTMLButtonElement;
const shuffleButton = document.getElementById('shuffle-button');
const volumeButton = document.getElementById('volume-button');
const recordButton = document.getElementById('record-button');
const volumeBar = document.getElementById('volume-bar') as HTMLInputElement;

player.getGain = () => volumeBar.valueAsNumber;
const updatePlayingState = () => {
  if (player.isPlaying) {
    playButton.classList.toggle('paused', true);
    audio.play();
  } else {
    playButton.classList.toggle('paused', false);
    audio.pause();
  }
};
player.onPlayingStateChange = updatePlayingState;
player.onLoadingStateChange = updateTrackClasses;
playButton.addEventListener('click', async () => {
  if (player.playlist.length === 0) return;
  
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
      player.repeat = RepeatMode.CONTINUOUS;
      repeatButton.classList.remove('repeat-one');
      repeatButton.classList.add('repeat-continuous');
      break;
    }
    case RepeatMode.CONTINUOUS: {
      player.repeat = RepeatMode.NONE;
      repeatButton.classList.remove('repeat-continuous');
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
  repeatButton.disabled = player.shuffle;
});
volumeButton.addEventListener('click', async () => {
  if (player.gain) {
    player.gain.gain.value = volumeBar.valueAsNumber;
  }
  player.muted = !player.muted;
  volumeButton.classList.toggle('muted', player.muted);
});

const updateRecordingState = () => {
  if (player.isRecording) {
    recordButton.classList.toggle('paused', false);
  } else {
    recordButton.classList.toggle('paused', true);
  }
};
player.onRecordingStateChange = updateRecordingState;
recordButton.addEventListener('click', async () => {
  if (player.isRecording) {
    player.pauseRecording();
  } else {
    player.startRecording();
  }
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
  if (exportPanel.style.visibility === 'visible') {
    exportPanel.style.visibility = 'hidden';
    exportPanel.style.opacity = '0';
  } else {
    exportPanel.style.visibility = 'visible';
    exportPanel.style.opacity = '1';
    const url = player.getExportUrl();
    exportUrlInput.value = url;
    // wait for panel to become visible before we can select the text field
    setTimeout(() => {
      exportUrlInput.select();
    }, 50);
  }
});
exportUrlInput.addEventListener('click', async () => {
  exportUrlInput.select();
});
copyButton.addEventListener('click', async () => {
  document.execCommand('copy');
  exportPanel.style.visibility = 'hidden';
  exportPanel.style.opacity = '0';
});

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
