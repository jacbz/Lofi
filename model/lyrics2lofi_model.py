import numpy as np
import torch
from torch import nn

from model.constants import *


class Lyrics2LofiModel(nn.Module):
    def __init__(self, device="cuda" if torch.cuda.is_available() else "cpu"):
        super(Lyrics2LofiModel, self).__init__()
        self.device = device
        self.encoder = Encoder(device)
        self.decoder = Decoder(device)
        self.mean_linear = nn.Linear(in_features=HIDDEN_SIZE, out_features=HIDDEN_SIZE)
        self.variance_linear = nn.Linear(in_features=HIDDEN_SIZE, out_features=HIDDEN_SIZE)

    def forward(self, input, num_chords=MAX_CHORD_LENGTH, sampling_rate_chords=0, sampling_rate_melodies=0,
                gt_chords=None, gt_melody=None):
        # encode
        h = self.encoder(input)

        # VAE
        mu = self.mean_linear(h)
        log_var = self.variance_linear(h)
        z = self.sample(mu, log_var)
        # compute the Kullback–Leibler divergence between a Gaussian and an uniform Gaussian
        kl = 0.5 * torch.mean(mu ** 2 + log_var.exp() - log_var - 1, dim=[0, 1])

        # decode
        if self.training:
            chord_outputs, melody_outputs, tempo_output, key_output, mode_output, valence_output, energy_output = \
                self.decoder(z, num_chords, sampling_rate_chords, sampling_rate_melodies, gt_chords, gt_melody)
        else:
            chord_outputs, melody_outputs, tempo_output, key_output, mode_output, valence_output, energy_output = \
                self.decoder(z, num_chords)

        return chord_outputs, melody_outputs, tempo_output, key_output, mode_output, valence_output, energy_output, kl

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
        self.encoder_lstm = nn.LSTM(input_size=BERT_EMBEDDING_LENGTH, hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS,
                                    bidirectional=True, batch_first=True)
        self.downsample = nn.Linear(in_features=2 * HIDDEN_SIZE, out_features=HIDDEN_SIZE)

    def forward(self, x):
        _, (h, _) = self.encoder_lstm(x)
        # downsample two directions
        return self.downsample(torch.cat((h[-1], h[-2]), dim=1))


class Decoder(nn.Module):
    def __init__(self, device):
        super(Decoder, self).__init__()
        self.device = device
        self.key_linear = nn.Sequential(
            nn.Linear(in_features=HIDDEN_SIZE, out_features=HIDDEN_SIZE),
            nn.ReLU(),
            nn.Linear(in_features=HIDDEN_SIZE, out_features=NUMBER_OF_KEYS),
        )
        self.mode_linear = nn.Sequential(
            nn.Linear(in_features=HIDDEN_SIZE, out_features=HIDDEN_SIZE),
            nn.ReLU(),
            nn.Linear(in_features=HIDDEN_SIZE, out_features=NUMBER_OF_MODES),
        )
        self.tempo_linear = nn.Sequential(
            nn.Linear(in_features=HIDDEN_SIZE, out_features=HIDDEN_SIZE),
            nn.ReLU(),
            nn.Linear(in_features=HIDDEN_SIZE, out_features=1),
        )
        self.valence_linear = nn.Sequential(
            nn.Linear(in_features=HIDDEN_SIZE, out_features=HIDDEN_SIZE),
            nn.ReLU(),
            nn.Linear(in_features=HIDDEN_SIZE, out_features=1),
        )
        self.energy_linear = nn.Sequential(
            nn.Linear(in_features=HIDDEN_SIZE, out_features=HIDDEN_SIZE),
            nn.ReLU(),
            nn.Linear(in_features=HIDDEN_SIZE, out_features=1),
        )
        self.tempo_embedding = nn.Linear(in_features=1, out_features=HIDDEN_SIZE)
        self.mode_embedding = nn.Linear(in_features=NUMBER_OF_MODES, out_features=HIDDEN_SIZE)
        self.valence_embedding = nn.Linear(in_features=1, out_features=HIDDEN_SIZE)
        self.energy_embedding = nn.Linear(in_features=1, out_features=HIDDEN_SIZE)
        self.downsample = nn.Linear(in_features=5 * HIDDEN_SIZE, out_features=HIDDEN_SIZE)

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

    def forward(self, z, num_chords, sampling_rate_chords=0, sampling_rate_melodies=0, gt_chords=None, gt_melody=None):
        tempo_output = self.tempo_linear(z)
        key_output = self.key_linear(z)
        mode_output = self.mode_linear(z)
        valence_output = self.valence_linear(z)
        energy_output = self.energy_linear(z)
        tempo_embedding = self.tempo_embedding(tempo_output)
        mode_embedding = self.mode_embedding(mode_output)
        valence_embedding = self.valence_embedding(valence_output)
        energy_embedding = self.energy_embedding(energy_output)
        z = self.downsample(torch.cat((z, tempo_embedding, mode_embedding, valence_embedding, energy_embedding), dim=1))

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
