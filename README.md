# LOFI

LOFI is a ML-supported lo-fi music generator. We trained a VAE model in [PyTorch](https://pytorch.org/) to generate chords, melodies and other musical parameters. The web client uses [Tone.js](https://tonejs.github.io/) to make a dusty Lofi track out of the model output.