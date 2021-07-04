import torch
from torch import nn
from constants import *

device = "cuda" if torch.cuda.is_available() else "cpu"


class Model(nn.Module):
    def __init__(self, hidden_size=200, num_layers=1):
        super(Model, self).__init__()
        self.encoder = Encoder(hidden_size, num_layers)
        self.decoder = Decoder(hidden_size)
        self.mean_linear = nn.Linear(in_features=hidden_size, out_features=hidden_size)
        self.variance_linear = nn.Linear(in_features=hidden_size, out_features=hidden_size)

    def forward(self, input, max_chords_length, gt_chords=None, gt_melody=None):
        # encode
        h = self.encoder(input)
        # add two directions together
        temp = h[-1] + h[-2]

        # VAE
        mu = self.mean_linear(temp)
        logvar = self.variance_linear(temp)
        z = self.sample(mu, logvar)
        # compute the Kullback–Leibler divergence between a Gaussian and an uniform Gaussian
        kl = 0.5 * torch.mean(mu ** 2 + logvar.exp() - logvar - 1, dim=[0, 1])

        # decode
        if self.training:
            chord_outputs, note_outputs, bpm_output, valence_output, energy_output = self.decoder(z, max_chords_length, gt_chords, gt_melody)
        else:
            chord_outputs, note_outputs, bpm_output, valence_output, energy_output = self.decoder(z, max_chords_length)

        return chord_outputs, note_outputs, bpm_output, valence_output, energy_output, kl

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
    def __init__(self, hidden_size, num_layers):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.rnn = nn.LSTM(input_size=BERT_EMBEDDING_LENGTH, hidden_size=hidden_size, num_layers=num_layers, bidirectional=True, batch_first=True)

    def forward(self, x):
        _, (h, _) = self.rnn(x)
        return h


class Decoder(nn.Module):
    def __init__(self, hidden_size):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.downsample = nn.Linear(in_features=4*hidden_size, out_features=hidden_size)
        self.bpm_linear = nn.Sequential(
            nn.Linear(in_features=hidden_size, out_features=hidden_size),
            nn.ReLU(),
            nn.Linear(in_features=hidden_size, out_features=1),
        )
        self.valence_linear = nn.Sequential(
            nn.Linear(in_features=hidden_size, out_features=hidden_size),
            nn.ReLU(),
            nn.Linear(in_features=hidden_size, out_features=1),
        )
        self.energy_linear = nn.Sequential(
            nn.Linear(in_features=hidden_size, out_features=hidden_size),
            nn.ReLU(),
            nn.Linear(in_features=hidden_size, out_features=1),
        )
        self.bpm_embedding = nn.Linear(in_features=1, out_features=hidden_size)
        self.valence_embedding = nn.Linear(in_features=1, out_features=hidden_size)
        self.energy_embedding = nn.Linear(in_features=1, out_features=hidden_size)
        self.chords_lstm = nn.LSTMCell(input_size=hidden_size * 1, hidden_size=hidden_size * 1)
        self.chord_embeddings = nn.Embedding(CHORD_PREDICTION_LENGTH, hidden_size)
        self.melody_embeddings = nn.Embedding(MELODY_PREDICTION_LENGTH, hidden_size)
        self.chord_layers = nn.Sequential(
            nn.Linear(in_features=hidden_size, out_features=hidden_size),
            nn.ReLU(),
            nn.Linear(in_features=hidden_size, out_features=CHORD_PREDICTION_LENGTH)
        )
        self.melody_lstm = nn.LSTMCell(input_size=hidden_size * 1, hidden_size=hidden_size * 1)
        self.melody_prediction = nn.Sequential(
            nn.Linear(in_features=hidden_size, out_features=hidden_size),
            nn.ReLU(),
            nn.Linear(in_features=hidden_size, out_features=MELODY_PREDICTION_LENGTH)
        )

    def forward(self, z, max_measure_length, gt_chords=None, gt_melody=None):
        bpm_output = self.bpm_linear(z)
        valence_output = self.valence_linear(z)
        energy_output = self.energy_linear(z)
        bpm_embedding = self.bpm_embedding(bpm_output)
        valence_embedding = self.valence_embedding(valence_output)
        energy_embedding = self.energy_embedding(energy_output)
        z = self.downsample(torch.cat((z, bpm_embedding, valence_embedding, energy_embedding), dim=1))

        hx_chords = torch.randn(z.shape[0], self.hidden_size * 1, device=device) # (batch, hidden_size)
        cx_chords = torch.randn(z.shape[0], self.hidden_size * 1, device=device)
        hx_melody = torch.randn(z.shape[0], self.hidden_size * 1, device=device) # (batch, hidden_size)
        cx_melody = torch.randn(z.shape[0], self.hidden_size * 1, device=device)

        hx_chords, cx_chords = self.chords_lstm(z, (hx_chords, cx_chords))
        chord_prediction = self.chord_layers(hx_chords)
        if gt_chords is not None:
            chord_embeddings = self.chord_embeddings(gt_chords[:,0])
        else:
            chord_embeddings = self.chord_embeddings(chord_prediction.argmax(dim=1))
        chord_outputs = [chord_prediction]
        hx_melody, cx_melody = self.melody_lstm(chord_embeddings, (hx_melody, cx_melody))
        melody_prediction = self.melody_prediction(hx_melody)
        note_outputs = [melody_prediction]
        for i in range(NOTES_PER_CHORD - 1):
            if gt_melody is not None:
                melody_embeddings = self.melody_embeddings(gt_melody[:,i])
            else:
                melody_embeddings = self.melody_embeddings(melody_prediction.argmax(dim=1))
            hx_melody, cx_melody = self.melody_lstm(melody_embeddings+chord_embeddings, (hx_melody, cx_melody))
            melody_prediction = self.melody_prediction(hx_melody)
            note_outputs.append(melody_prediction)

        # stop when reaching max length
        for i in range(1, max_measure_length * CHORD_DISCRETIZATION_LENGTH):
            hx_chords, cx_chords = self.chords_lstm(chord_embeddings, (hx_chords, cx_chords))
            chord_prediction = self.chord_layers(hx_chords)
            if gt_chords is not None:
                chord_embeddings = self.chord_embeddings(gt_chords[:,i])
            else:
                chord_embeddings = self.chord_embeddings(chord_prediction.argmax(dim=1))
            chord_outputs.append(chord_prediction)
            for j in range(NOTES_PER_CHORD):
                if gt_melody is not None:
                    melody_embeddings = self.melody_embeddings(gt_melody[:, i*NOTES_PER_CHORD+j-1])
                else:
                    melody_embeddings = self.melody_embeddings(melody_prediction.argmax(dim=1))
                hx_melody, cx_melody = self.melody_lstm(melody_embeddings + chord_embeddings, (hx_melody, cx_melody))
                melody_prediction = self.melody_prediction(hx_melody)
                note_outputs.append(melody_prediction)

        chord_outputs = torch.stack(chord_outputs, dim=1)
        note_outputs = torch.stack(note_outputs, dim=1)
        return chord_outputs, note_outputs, bpm_output, valence_output, energy_output
