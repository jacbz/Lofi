# Datasets

The training dataset is synthesized using multiple sources.

* Chords and melodies are obtained from [Hooktheory](https://www.hooktheory.com/)
* The lyrics are obtained by scraping Google or Musixmatch
* Audio features are obtained using the [Spotify API](https://developer.spotify.com/documentation/web-api/)



To build the training set and add lyrics and audio features:

1. Download the Hooktheory dataset from [this](https://github.com/wayne391/lead-sheet-dataset) repo and copy the `event` folder into this directory, named `hooktheory`
2. Register application at the [Spotify Developer Dashboard](https://developer.spotify.com/dashboard)
3. Write client id into the file `spotify_client_id` and secret into `spotify_client_secret`
4. Set `add_lyrics` (true/false), `add_spotify` (true/false) and `lyrics_provider` (google/musixmatch) inside `prepocessor.py`
5. Run `python prepocessor.py`
6. The dataset will be built into the folder `processed`