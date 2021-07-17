# LOFI Server

This is a simple [Flask](https://palletsprojects.com/p/flask/) server which offers two API endpoints:

* `/decode`: This endpoints takes a number array as parameter `input`. The number array must have the same dimension as the latent space. The server delivers a lo-fi track by running the input through the Lofi2Lofi decoder.
* `/predict`: This endpoint takes a string as parameter `input` and delivers a lo-fi track by running the input through Lyrics2Lofi.

You need to save the two checkpoints in `checkpoints/lofi2lofi_decoder.pth` and `checkpoints/lyrics2lofi.pth` respectively.