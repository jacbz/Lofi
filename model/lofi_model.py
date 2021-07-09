import numpy as np
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from model.constants import *


class Model(nn.Module):
    def __init__(self, hidden_size=400, num_layers=1, device="cuda" if torch.cuda.is_available() else "cpu"):
        super(Model, self).__init__()
        self.device = device
        self.encoder = Encoder(hidden_size, num_layers, device)
        self.decoder = Decoder(hidden_size, device)
        self.mean_linear = nn.Linear(in_features=hidden_size, out_features=hidden_size)
        self.variance_linear = nn.Linear(in_features=hidden_size, out_features=hidden_size)
        self.hidden_size = hidden_size

    def forward(self, gt_chords, gt_melodies, batch_num_chords, num_chords, sampling_rate_chords=0, sampling_rate_melodies=0):
        # encode
        h = self.encoder(gt_chords, gt_melodies, batch_num_chords)
        # add two directions together
        temp = h[-1] + h[-2]

        # VAE
        mu = self.mean_linear(temp)
        log_var = self.variance_linear(temp)
        z = self.sample(mu, log_var)
        # compute the Kullback–Leibler divergence between a Gaussian and an uniform Gaussian
        kl = 0.5 * torch.mean(mu ** 2 + log_var.exp() - log_var - 1, dim=[0, 1])

        # decode
        if self.training:
            chord_outputs, melody_outputs = \
                self.decoder(z, num_chords, sampling_rate_chords, sampling_rate_melodies, gt_chords, gt_melodies)
        else:
            chord_outputs, melody_outputs = \
                self.decoder(z, num_chords)

        return chord_outputs, melody_outputs, kl

    # reparameterization trick:
    # because backpropagation cannot flow through a random node, we introduce a new parameter that allows us to
    # reparameterize z in a way that allows backprop to flow through the deterministic nodes
    # https://stats.stackexchange.com/questions/199605/how-does-the-reparameterization-trick-for-vaes-work-and-why-is-it-important
    def sample(self, mu, logvar):
        if self.training:
            return mu + torch.randn_like(mu) * (logvar / 2).exp()
        else:
            return mu

    def generate(self):
        mu = torch.randn(1,self.hidden_size)
        return self.decoder(mu, MAX_CHORD_LENGTH)

    def interpolate(self, chords1, chords2):
        pass


class Encoder(nn.Module):
    def __init__(self, hidden_size, num_layers, device):
        super(Encoder, self).__init__()
        self.device = device
        self.hidden_size = hidden_size
        self.chord_embeddings = nn.Embedding(CHORD_PREDICTION_LENGTH, hidden_size)
        self.chords_lstm = nn.LSTM(input_size=hidden_size, hidden_size=hidden_size, num_layers=num_layers, bidirectional=True, batch_first=True)

        self.melody_embeddings = nn.Embedding(MELODY_PREDICTION_LENGTH, hidden_size)
        self.melody_lstm = nn.LSTM(input_size=hidden_size, hidden_size=hidden_size, num_layers=num_layers, bidirectional=True, batch_first=True)

    def forward(self, chords, melodies, batch_num_chords):
        chord_embeddings = self.chord_embeddings(chords)
        chords_input = pack_padded_sequence(chord_embeddings, batch_num_chords, batch_first=True, enforce_sorted=False)

        chords_out, (h_chords, _) = self.chords_lstm(chords_input)
        chords_out_repeated = pad_packed_sequence(chords_out, batch_first=True)[0].repeat_interleave( NOTES_PER_CHORD, 1)
        chords_out_repeated = chords_out_repeated[:,:,:self.hidden_size] + chords_out_repeated[:,:,self.hidden_size:]

        # add two directions together
        melody_embeddings = self.melody_embeddings(melodies) + chords_out_repeated
        melody_input = pack_padded_sequence(melody_embeddings, batch_num_chords * NOTES_PER_CHORD, batch_first=True, enforce_sorted=False)

        _, (h_melodies, _) = self.melody_lstm(melody_input)
        return h_melodies


class Decoder(nn.Module):
    def __init__(self, hidden_size, device):
        super(Decoder, self).__init__()
        self.device = device
        self.hidden_size = hidden_size

        self.chords_lstm = nn.LSTMCell(input_size=hidden_size * 1, hidden_size=hidden_size * 1)
        self.chord_embeddings = nn.Embedding(CHORD_PREDICTION_LENGTH, hidden_size)
        self.chord_layers = nn.Sequential(
            nn.Linear(in_features=hidden_size, out_features=hidden_size),
            nn.ReLU(),
            nn.Linear(in_features=hidden_size, out_features=CHORD_PREDICTION_LENGTH)
        )

        self.melody_embeddings = nn.Embedding(MELODY_PREDICTION_LENGTH, hidden_size)
        self.melody_lstm = nn.LSTMCell(input_size=hidden_size * 1, hidden_size=hidden_size * 1)
        self.melody_prediction = nn.Sequential(
            nn.Linear(in_features=hidden_size, out_features=hidden_size),
            nn.ReLU(),
            nn.Linear(in_features=hidden_size, out_features=MELODY_PREDICTION_LENGTH)
        )

    def forward(self, z, num_chords, sampling_rate_chords=0, sampling_rate_melodies=0, gt_chords=None, gt_melody=None):
        # initialize hidden states and cell states randomly
        hx_chords = torch.randn(z.shape[0], self.hidden_size * 1, device=self.device)
        cx_chords = torch.randn(z.shape[0], self.hidden_size * 1, device=self.device)
        hx_melody = torch.randn(z.shape[0], self.hidden_size * 1, device=self.device)
        cx_melody = torch.randn(z.shape[0], self.hidden_size * 1, device=self.device)

        chord_outputs = []
        note_outputs = []

        # the chord LSTM input at first only consists of z
        # after the first iteration, we use the chord embeddings
        chords_lstm_input = z
        melody_embeddings = 0  # these will be set in the very first iteration

        for i in range(num_chords):
            hx_chords, cx_chords = self.chords_lstm(chords_lstm_input, (hx_chords, cx_chords))
            chord_prediction = self.chord_layers(hx_chords)

            # perform teacher forcing during training
            perform_teacher_forcing_chords = bool(
                np.random.choice(2, 1, p=[1 - sampling_rate_chords, sampling_rate_chords])[0])
            if gt_chords is not None and perform_teacher_forcing_chords:
                chord_embeddings = self.chord_embeddings(gt_chords[:, i])
            else:
                chord_embeddings = self.chord_embeddings(chord_prediction.argmax(dim=1))

            chords_lstm_input = chord_embeddings
            chord_outputs.append(chord_prediction)

            # the melody LSTM input at first only includes the chord embeddings
            # after the first iteration, the input also includes the melody embeddings of the notes up to that point
            melody_lstm_input = melody_embeddings + chord_embeddings
            for j in range(NOTES_PER_CHORD):
                hx_melody, cx_melody = self.melody_lstm(melody_lstm_input, (hx_melody, cx_melody))
                melody_prediction = self.melody_prediction(hx_melody)
                note_outputs.append(melody_prediction)
                # perform teacher forcing during training
                perform_teacher_forcing = bool(
                    np.random.choice(2, 1, p=[1 - sampling_rate_melodies, sampling_rate_melodies])[0])
                if gt_melody is not None and perform_teacher_forcing:
                    melody_embeddings = self.melody_embeddings(gt_melody[:, i * NOTES_PER_CHORD + j])
                else:
                    melody_embeddings = self.melody_embeddings(melody_prediction.argmax(dim=1))
                melody_lstm_input = melody_embeddings + chord_embeddings

        chord_outputs = torch.stack(chord_outputs, dim=1)
        if len(note_outputs) > 0:
            note_outputs = torch.stack(note_outputs, dim=1)
        return chord_outputs, note_outputs
