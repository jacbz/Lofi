[![Build][build-shield]][build-url]
[![Stargazers][stars-shield]][stars-url]
[![Forks][forks-shield]][forks-url]
[![Issues][issues-shield]][issues-url]
[![Contributors][contributors-shield]][contributors-url]
[![Apache License][license-shield]][license-url]


<!-- PROJECT LOGO -->
<br />
<div align="center">
  <a href="https://lofi.jacobzhang.de/?default">
    <img src="https://repository-images.githubusercontent.com/377117802/d55ba858-636f-4c44-9195-94971754fec0" width="400px"/>
  </a>

  <h3 align="center">Lofi</h3>

  <p align="center">
    A ML-supported lo-fi music generator.
    <br />
    <a href="https://lofi.jacobzhang.de/?default"><strong>Explore »</strong></a>
  </p>
</div>

## About
We trained a VAE model in [PyTorch](https://pytorch.org/) to represent a lo-fi track as a vector of 100 features. A lo-fi track consists of chords, melodies, and other musical parameters. The web client uses [Tone.js](https://tonejs.github.io/) to make a dusty lo-fi track out of these parameters.

<div align="center">
  <a href="https://lofi.jacobzhang.de/?default">
    <img src="https://i.imgur.com/cxFsYPm.jpg" width="800px"/>
  </a>
</div>

## Architecture

* **Client**: The client is written in TypeScript and built with Webpack. It uses Tone.js to generate music.
* **Model**: The model is implemented in PyTorch. We synthesized various datasets, including Hooktheory and Spotify.
* **Server**: The server is a basic Flask instance that deploys the trained model checkpoint. The client communicates with the server using a REST API.

<img src="https://i.imgur.com/j70305Y.png" width="600px"/>

## Setup
If you only want to tinker around with the client, you will only need the `client` folder. This will use the project's server as the backend.

If you want to deploy your own model, you can either train your own model (see the instructions in the `model`) or download the pre-trained checkpoint from [here](https://github.com/jacbz/Lofi/files/7519187/checkpoints.zip). Once you have deployed the server, change the server address inside `client\src\api.ts`.

### Set up the client
1. Install [node.js](https://nodejs.org/en/) LTS.
1. Navigate to the client folder and run `npm install` to install the dependencies.
2. Run `npm run serve` to develop or `npm run build` to build a distributable.

By default, this uses the project's server as the backend. You can also train your own model and deploy your own server.

### Train your own model
See the [model](model) folder for details. Once you have trained your model, put the checkpoint in the `checkpoints` folder.

### Deploy your own server
See the [server](server) folder for details. You can use the provided Dockerfile. Don't forget to change the API url in the client.

## Contributing
Contributions are what make the open source community such an amazing place to learn, inspire, and create. Any contributions you make are greatly appreciated.

If you have a suggestion that would make this better, please fork the repo and create a pull request. You can also simply open an issue with the tag "enhancement".

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## Acknowledgments

Big thanks to [ZOOPRA UG](https://www.zoopra.de/) for hosting the server!

<a href="https://www.zoopra.de/">
  <img src="https://www.zoopra.de/wp-content/uploads/2020/11/logo_breit_800x200_black.png" href="https://www.zoopra.de" width="200px"/>
</a>



<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[build-shield]: https://img.shields.io/github/actions/workflow/status/jacbz/Lofi/client.yml?style=for-the-badge
[build-url]: https://github.com/jacbz/Lofi/actions
[contributors-shield]: https://img.shields.io/github/contributors/jacbz/Lofi?style=for-the-badge
[contributors-url]: https://github.com/jacbz/Lofi/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/jacbz/Lofi?style=for-the-badge
[forks-url]: https://github.com/jacbz/Lofi/network/members
[stars-shield]: https://img.shields.io/github/stars/jacbz/Lofi?style=for-the-badge
[stars-url]: https://github.com/jacbz/Lofi/stargazers
[issues-shield]: https://img.shields.io/github/issues/jacbz/Lofi?style=for-the-badge
[issues-url]: https://github.com/jacbz/Lofi/issues
[license-shield]: https://img.shields.io/github/license/jacbz/Lofi?style=for-the-badge
[license-url]: https://github.com/jacbz/Lofi/blob/master/LICENSE.txt
