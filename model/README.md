# LOFI Model

Lo-fi music has rather simple characteristics (short loops of chord progressions, simple melodies, no dynamics, etc.) which makes it an easy target for computational music generation. We obtained a dataset containing thousands of songs, each sample containing chord progressions, melodies, and other musical parameters, and trained two VAE models (Lofi2Lofi and Lyrics2Lofi) to generate a latent space of musical parameters to sample from. These musical parameters are:

* **Chords**: an integer sequence of a chord progression in Roman numeral notation (0-8; 0=rest; 8=end)

* **Melodies**: an integer sequence of eight notes for each chord, by scale degree over two octaves (0-15; 0=rest)

* **Tempo**: a continuous value between [0, 1] that indicates the tempo, can be scaled to BPM

* **Key**: an integer between 1 and 12 denoting the musical key, by chromatic scale degree

* **Mode**: an integer between 1 and 7 corresponding to one of the seven Greek modes

* **Valence**: a continuous value between [0, 1] that denotes musical positiveness

* **Energy**: a continuous value between [0, 1] that denotes a perceptual measure of intensity and activity

A sample can thus be represented in JSON as such:

```json
{
  "title": "#338934052871945450670672",
  "key": 3,
  "mode": 1,
  "bpm": 75,
  "energy": 0.501,
  "valence": 0.381,
  "chords": [6, 7, 1, 4, 5, 1],
  "melodies": [
    [0, 6, 6, 6, 5, 6, 5, 6],
    [5, 6, 2, 0, 2, 0, 2, 0],
    [0, 5, 0, 5, 0, 5, 0, 5],
    [0, 6, 6, 6, 6, 6, 6, 6],
    [6, 5, 5, 5, 5, 5, 5, 2],
    [0, 5, 0, 0, 0, 0, 0, 0]
  ]
}
```

## Models

We trained two VAEs, Lofi2Lofi and Lyrics2Lofi.

### Lofi2Lofi

Lofi2Lofi is a symmetrical VAE. Each dataset sample is encoded in the same format as the output.

![](https://svgshare.com/i/ZF5.svg)

The architecture of the decoder is easier to look at if we unroll the LSTMs:

![](https://svgshare.com/i/ZEy.svg)

This architecture ensures that melodies are conditioned on the chord and each note is conditioned on the previous notes. In music, context is very important, and we hope to reflect that in our model.

### Lyrics2Lofi

Lyrics2Lofi is an asymmetrical VAE which takes lyrics as input. We initially hoped to be able to turn input text into lo-fi music, but preliminary results show that text embeddings are simply not able to provide enough information about the music itself, leading to poor validation performance.

![](https://svgshare.com/i/ZCo.svg)

## Running the model

First, follow the instructions in the `dataset` folder to build the dataset.

To run Lofi2Lofi:

1. Run `lofi2lofi_train.py`

To run Lyrics2Lofi:

1. Run `make_embeddings` inside `embeddings.py` to build the `embeddings.npy` file.
2. Run `lyrics2lofi_train.py`

