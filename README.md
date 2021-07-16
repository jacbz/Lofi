# LOFI

LOFI is a ML-supported lo-fi music generator. We trained a VAE model in [PyTorch](https://pytorch.org/) to represent a lo-fi track as a vector of 100 features. A lo-fi track consists of chords, melodies and other musical parameters. The web client uses [Tone.js](https://tonejs.github.io/) to make a dusty lo-fi track out of these parameters.

<p align="center">
  <img src="https://repository-images.githubusercontent.com/377117802/d55ba858-636f-4c44-9195-94971754fec0" width="400px"/>
</p>
**Click [here](http://lofi.jacobzhang.de/?eJyFVltu3DAMvIv7KxTiS6JylcV+tInRBk2aIg0KFEXuXtqWvHqsN6AdZymJFkfDoU//prfHt6d5ups+QdQUiCSEhICMGhIHZp7c9GP+O92Rm55fHmwquOnrr+fpTu2f+ef8+s0G/efI6KY/X57mn/fz8jtgdNP995fXh9/T3QlccOLAsd1ytkjz08vD47wMnZLT2s7uFFxj2bOMgl/+gD3NqQcLl/HLZe51VTtTbCubxcXO53dXYwFRVNR7oESQEpCQiGjBAgoWIWMRscYCVRoskKHBQlYc9meHh5q/snWzvUe7JNtU63zb9I8jXot3iF792zvfoYeRvffs1VAEDBACR6G0Myn2TIpSo2cUbNDzqSMSbTSy50Konk5jaiOdyBXz6/4vcy450rqehmPwxao4Fw84PXi7X2+/xvG19ehpiFGFQQgoKAuyxxiloCdDHaYavcAteqTawLcAuLEvDNBF2/xSp2FlJucESzLsaE2n3bw7eRvxH8Dih1XX4tyCBTz4aNKkmGLyPggLYyLcSxILLlJYFWtcQLSrSexoxRkXyOhsHh6qcxSdWFmtX1Y55dqLpvL1xVkCxiuexWcp5stW47IWW9/lXc0I1iOh20Pt74t8mJeDX3HfHmmP0uQAWEEjSdKomgCFWXeGDyepvmG4h+YkpWF4zLoaF2kYtHXjFhTxXwsWDniYWW8evsHVEifszLnMwdryu+CQ4YTGZybruzGRdWFTgBCpYvjYdRpcEHqGt5VPxkijgt2bAuBVbKSh3dh39i6SPcuavrVe619HDeS4I33s6YTTlJIYTR1CChJUSGP0SqNC7H2HGvxi13c8VfjhyivJfdvfaDoFBarNPDjwYdSSZk1GqsQ9jtPXbqeczEgQWBkUraFAghRQ0g1cGuXshZNq4QyZTGGTzA4Wq/5Un/QmEddJlb8lcoccPk98+arx1awevlp2qp6M2ydj9Yrz+/k/KIyMcg==) for a pre-generated lo-fi playlist!**

## Architecture

* **Client**: The client is written in TypeScript and built with Webpack. It uses Tone.js to generate music.
* **Model**: The model is implemented in PyTorch. We synthesized various datasets, including Hooktheory and Spotify.
* **Server**: The server is a basic Flask instance which deploys the trained model checkpoint. The client communicates with the Server using a REST API.

![](https://i.imgur.com/j70305Y.png)