# LOFI

![](https://github.com/jacbz/lofi/actions/workflows/client.yml/badge.svg)

LOFI is a ML-supported lo-fi music generator. We trained a VAE model in [PyTorch](https://pytorch.org/) to represent a lo-fi track as a vector of 100 features. A lo-fi track consists of chords, melodies, and other musical parameters. The web client uses [Tone.js](https://tonejs.github.io/) to make a dusty lo-fi track out of these parameters.

<p align="center">
  <img src="https://repository-images.githubusercontent.com/377117802/d55ba858-636f-4c44-9195-94971754fec0" width="400px"/>
</p>

Click [here](http://lofi.jacobzhang.de/?default) for a pre-generated lo-fi playlist!

## Architecture

* **Client**: The client is written in TypeScript and built with Webpack. It uses Tone.js to generate music.
* **Model**: The model is implemented in PyTorch. We synthesized various datasets, including Hooktheory and Spotify.
* **Server**: The server is a basic Flask instance that deploys the trained model checkpoint. The client communicates with the server using a REST API.

## Setup
If you only want to tinker around with the client, you will only need the `client` folder. This will use the project's server as the backend.

If you want to deploy your own model, you can either train your own model (see the instructions in the `model`) or download the pre-trained checkpoint from [here](https://github.com/jacbz/Lofi/files/7519187/checkpoints.zip). Once you have deployed the server, change the server address inside `client\src\api.ts`.