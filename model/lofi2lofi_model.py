from hashlib import md5

import numpy as np
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence

from model.constants import *


class Lofi2LofiModel(nn.Module):
    def __init__(self, device="cuda" if torch.cuda.is_available() else "cpu"):
        super(Lofi2LofiModel, self).__init__()
        self.device = device
        self.encoder = Encoder(device)
        self.decoder = Decoder(device)
        self.mean_linear = nn.Linear(in_features=HIDDEN_SIZE, out_features=HIDDEN_SIZE)
        self.variance_linear = nn.Linear(in_features=HIDDEN_SIZE, out_features=HIDDEN_SIZE)

    def forward(self, gt_chords, gt_melodies, gt_tempo, gt_key, gt_mode, gt_valence, gt_energy, batch_num_chords,
                num_chords, sampling_rate_chords=0, sampling_rate_melodies=0):
        # encode
        h = self.encoder(gt_chords, gt_melodies, gt_tempo, gt_key, gt_mode, gt_valence, gt_energy, batch_num_chords)
        # VAE
        mu = self.mean_linear(h)
        log_var = self.variance_linear(h)
        z = self.sample(mu, log_var)
        # compute the Kullback–Leibler divergence between a Gaussian and an uniform Gaussian
        kl = 0.5 * torch.mean(mu ** 2 + log_var.exp() - log_var - 1, dim=[0, 1])

        # decode
        if self.training:
            chord_outputs, melody_outputs, tempo, key, mode, valence, energy = \
                self.decoder(z, num_chords, sampling_rate_chords, sampling_rate_melodies, gt_chords, gt_melodies)
        else:
            chord_outputs, melody_outputs, tempo, key, mode, valence, energy = \
                self.decoder(z, num_chords)

        return chord_outputs, melody_outputs, tempo, key, mode, valence, energy, kl

    # reparameterization trick:
    # because backpropagation cannot flow through a random node, we introduce a new parameter that allows us to
    # reparameterize z in a way that allows backprop to flow through the deterministic nodes
    # https://stats.stackexchange.com/questions/199605/how-does-the-reparameterization-trick-for-vaes-work-and-why-is-it-important
    def sample(self, mu, logvar):
        if self.training:
            return mu + torch.randn_like(mu) * (logvar / 2).exp()
        else:
            return mu


class Encoder(nn.Module):
    def __init__(self, device):
        super(Encoder, self).__init__()
        self.device = device
        self.chord_embeddings = nn.Embedding(num_embeddings=CHORD_PREDICTION_LENGTH, embedding_dim=HIDDEN_SIZE)
        self.chords_lstm = nn.LSTM(input_size=HIDDEN_SIZE, hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS,
                                   bidirectional=True, batch_first=True)

        self.melody_embeddings = nn.Embedding(num_embeddings=MELODY_PREDICTION_LENGTH, embedding_dim=HIDDEN_SIZE)
        self.melody_lstm = nn.LSTM(input_size=HIDDEN_SIZE, hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS,
                                   bidirectional=True, batch_first=True)

        self.tempo_embedding = nn.Linear(in_features=1, out_features=HIDDEN_SIZE2)
        self.key_embedding = nn.Embedding(num_embeddings=NUMBER_OF_KEYS, embedding_dim=HIDDEN_SIZE2)
        self.mode_embedding = nn.Embedding(num_embeddings=NUMBER_OF_MODES, embedding_dim=HIDDEN_SIZE2)
        self.valence_embedding = nn.Linear(in_features=1, out_features=HIDDEN_SIZE2)
        self.energy_embedding = nn.Linear(in_features=1, out_features=HIDDEN_SIZE2)

        self.downsample = nn.Linear(in_features=4 * HIDDEN_SIZE + 5 * HIDDEN_SIZE2, out_features=HIDDEN_SIZE)

    def forward(self, chords, melodies, tempo, key, mode, valence, energy, batch_num_chords):
        chord_embeddings = self.chord_embeddings(chords)
        chords_input = pack_padded_sequence(chord_embeddings, batch_num_chords, batch_first=True, enforce_sorted=False)
        chords_out, (h_chords, _) = self.chords_lstm(chords_input)

        melody_embeddings = self.melody_embeddings(melodies)
        melody_input = pack_padded_sequence(melody_embeddings, batch_num_chords * NOTES_PER_CHORD, batch_first=True,
                                            enforce_sorted=False)
        _, (h_melodies, _) = self.melody_lstm(melody_input)

        tempo_embedding = self.tempo_embedding(tempo.unsqueeze(1).float())
        key_embedding = self.key_embedding(key)
        mode_embedding = self.mode_embedding(mode)
        valence_embedding = self.valence_embedding(valence.unsqueeze(1).float())
        energy_embedding = self.energy_embedding(energy.unsqueeze(1).float())

        h_concatenated = torch.cat((h_chords[-1], h_chords[-2], h_melodies[-1], h_melodies[-2]), dim=1)
        return self.downsample(torch.cat(
            (h_concatenated, tempo_embedding, key_embedding, mode_embedding, valence_embedding, energy_embedding),
            dim=1))


class Decoder(nn.Module):
    def __init__(self, device):
        super(Decoder, self).__init__()
        self.device = device

        self.chords_lstm = nn.LSTMCell(input_size=HIDDEN_SIZE * 1, hidden_size=HIDDEN_SIZE * 1)
        self.chord_embeddings = nn.Embedding(num_embeddings=CHORD_PREDICTION_LENGTH, embedding_dim=HIDDEN_SIZE)
        self.chord_prediction = nn.Sequential(
            nn.Linear(in_features=HIDDEN_SIZE, out_features=HIDDEN_SIZE),
            nn.ReLU(),
            nn.Linear(in_features=HIDDEN_SIZE, out_features=CHORD_PREDICTION_LENGTH)
        )
        self.chord_embedding_downsample = nn.Linear(in_features=2 * HIDDEN_SIZE, out_features=HIDDEN_SIZE)

        self.melody_embeddings = nn.Embedding(num_embeddings=MELODY_PREDICTION_LENGTH, embedding_dim=HIDDEN_SIZE)
        self.melody_lstm = nn.LSTMCell(input_size=HIDDEN_SIZE * 1, hidden_size=HIDDEN_SIZE * 1)
        self.melody_prediction = nn.Sequential(
            nn.Linear(in_features=HIDDEN_SIZE, out_features=HIDDEN_SIZE),
            nn.ReLU(),
            nn.Linear(in_features=HIDDEN_SIZE, out_features=MELODY_PREDICTION_LENGTH)
        )
        self.melody_embedding_downsample = nn.Linear(in_features=3 * HIDDEN_SIZE, out_features=HIDDEN_SIZE)

        self.key_linear = nn.Sequential(
            nn.Linear(in_features=HIDDEN_SIZE, out_features=HIDDEN_SIZE2),
            nn.ReLU(),
            nn.Linear(in_features=HIDDEN_SIZE2, out_features=NUMBER_OF_KEYS),
        )
        self.mode_linear = nn.Sequential(
            nn.Linear(in_features=HIDDEN_SIZE, out_features=HIDDEN_SIZE2),
            nn.ReLU(),
            nn.Linear(in_features=HIDDEN_SIZE2, out_features=NUMBER_OF_MODES),
        )
        self.tempo_linear = nn.Sequential(
            nn.Linear(in_features=HIDDEN_SIZE, out_features=HIDDEN_SIZE2),
            nn.ReLU(),
            nn.Linear(in_features=HIDDEN_SIZE2, out_features=1),
        )
        self.valence_linear = nn.Sequential(
            nn.Linear(in_features=HIDDEN_SIZE, out_features=HIDDEN_SIZE2),
            nn.ReLU(),
            nn.Linear(in_features=HIDDEN_SIZE2, out_features=1),
        )
        self.energy_linear = nn.Sequential(
            nn.Linear(in_features=HIDDEN_SIZE, out_features=HIDDEN_SIZE2),
            nn.ReLU(),
            nn.Linear(in_features=HIDDEN_SIZE2, out_features=1),
        )

    def decode(self, mu):
        # create a hash for vector mu
        hash = ""
        # first 20 characters are each sampled from 5 entries
        for i in range(0, 100, 5):
            hash += str((mu[0][i:i + 1].abs().sum() * 587).int().item())[-1]
        # last 4 characters are the beginning of the MD5 hash of the whole vector
        hash2 = int(md5(mu.numpy()).hexdigest(), 16)
        hash = f"#{hash}{hash2}"[:25]
        return hash, self(mu, MAX_CHORD_LENGTH)

    def forward(self, z, num_chords=MAX_CHORD_LENGTH, sampling_rate_chords=0, sampling_rate_melodies=0, gt_chords=None,
                gt_melody=None):
        tempo_output = self.tempo_linear(z)
        key_output = self.key_linear(z)
        mode_output = self.mode_linear(z)
        valence_output = self.valence_linear(z)
        energy_output = self.energy_linear(z)

        batch_size = z.shape[0]
        # initialize hidden states and cell states
        hx_chords = torch.zeros(batch_size, HIDDEN_SIZE, device=self.device)
        cx_chords = torch.zeros(batch_size, HIDDEN_SIZE, device=self.device)
        hx_melody = torch.zeros(batch_size, HIDDEN_SIZE, device=self.device)
        cx_melody = torch.zeros(batch_size, HIDDEN_SIZE, device=self.device)

        chord_outputs = []
        melody_outputs = []

        # the chord LSTM input at first only consists of z
        # after the first iteration, we use the chord embeddings
        chord_embeddings = z
        melody_embeddings = None  # these will be set in the very first iteration

        for i in range(num_chords):
            hx_chords, cx_chords = self.chords_lstm(chord_embeddings, (hx_chords, cx_chords))
            chord_prediction = self.chord_prediction(hx_chords)
            chord_outputs.append(chord_prediction)

            # perform teacher forcing during training
            perform_teacher_forcing_chords = bool(
                np.random.choice(2, 1, p=[1 - sampling_rate_chords, sampling_rate_chords])[0])
            if gt_chords is not None and perform_teacher_forcing_chords:
                chord_embeddings = self.chord_embeddings(gt_chords[:, i])
            else:
                chord_embeddings = self.chord_embeddings(chord_prediction.argmax(dim=1))

            # let z influence the chord embedding
            chord_embeddings = self.chord_embedding_downsample(torch.cat((chord_embeddings, z), dim=1))

            # the melody LSTM input at first only includes the chord embeddings
            # after the first iteration, the input also includes the melody embeddings of the notes up to that point
            if melody_embeddings is None:
                melody_embeddings = chord_embeddings
            for j in range(NOTES_PER_CHORD):
                hx_melody, cx_melody = self.melody_lstm(melody_embeddings, (hx_melody, cx_melody))
                melody_prediction = self.melody_prediction(hx_melody)
                melody_outputs.append(melody_prediction)
                # perform teacher forcing during training
                perform_teacher_forcing = bool(
                    np.random.choice(2, 1, p=[1 - sampling_rate_melodies, sampling_rate_melodies])[0])
                if gt_melody is not None and perform_teacher_forcing:
                    melody_embeddings = self.melody_embeddings(gt_melody[:, i * NOTES_PER_CHORD + j])
                else:
                    melody_embeddings = self.melody_embeddings(melody_prediction.argmax(dim=1))
                melody_embeddings = self.melody_embedding_downsample(
                    torch.cat((melody_embeddings, chord_embeddings, z), dim=1))

        chord_outputs = torch.stack(chord_outputs, dim=1)
        melody_outputs = torch.stack(melody_outputs, dim=1)

        return chord_outputs, melody_outputs, tempo_output, key_output, mode_output, valence_output, energy_output
