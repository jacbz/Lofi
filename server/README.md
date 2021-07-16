# LOFI Server

This is a simple [Flask](https://palletsprojects.com/p/flask/) server which offers two API endpoints:

* `/generate`: This endpoint takes no input and delivers a lo-fi track sampled from the latent distribution.
* `/decode`: This endpoints takes a number array as parameter `input`. The number array must have the same dimension as the latent space. The server delivers a lo-fi track by running the input through the decoder.

