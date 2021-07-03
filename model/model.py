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

    def forward(self, input, max_chords_length, gt_chords=None, gt_melody=None):
        # pred, (hidden states, cell states)
        h = self.encoder(input)
        temp = h[-1] + h[-2]
        # add two directions together

        # VAE
        mu = self.mean_linear(temp)
        logvar = self.variance_linear(temp)
        z = self.sample(mu, logvar)
        bpm = self.bpm_linear(z)
        valence = self.valence_linear(z)
        energy = self.energy_linear(z)
        bpm_embedding = self.bpm_embedding(bpm)
        valence_embedding = self.valence_embedding(valence)
        energy_embedding = self.energy_embedding(energy)
        z = self.downsample(torch.cat((z, bpm_embedding, valence_embedding, energy_embedding), dim=1))
        # make chord prediction
        chords_pred = self.decoder(z, max_chords_length, gt_chords, gt_melody)

        # compute the Kullback–Leibler divergence between a Gaussian and an uniform Gaussian
        KL = 0.5 * torch.mean(mu ** 2 + logvar.exp() - logvar - 1, dim=[0, 1])

        return chords_pred, KL, bpm, valence, energy

    # reparameterization trick
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
        )
        self.melody_note_prediction = nn.Linear(in_features=hidden_size, out_features=MELODY_PREDICTION_LENGTH)


    def forward(self, x, max_chords_length, gt_chords=None, gt_melody=None):
        hx_chords = torch.randn(x.shape[0], self.hidden_size * 1, device=device) # (batch, hidden_size)
        cx_chords = torch.randn(x.shape[0], self.hidden_size * 1, device=device)
        hx_melody = torch.randn(x.shape[0], self.hidden_size * 1, device=device) # (batch, hidden_size)
        cx_melody = torch.randn(x.shape[0], self.hidden_size * 1, device=device)

        hx_chords, cx_chords = self.chords_lstm(x, (hx_chords, cx_chords))
        chord_prediction = self.chord_layers(hx_chords)
        if gt_chords is not None:
            chord_embeddings = self.chord_embeddings(gt_chords[:,0])
        else:
            chord_embeddings = self.chord_embeddings(chord_prediction.argmax(dim=1))

        chord_outputs = [chord_prediction]

        hx_melody, cx_melody = self.melody_lstm(chord_embeddings, (hx_melody, cx_melody))
        md = self.melody_prediction(hx_melody)

        melody_note_prediction = self.melody_note_prediction(md)

        note_outputs = [melody_note_prediction]

        for i in range(MELODY_DISCRETIZATION_LENGTH-1):
            if gt_melody is not None:
                melody_embeddings = self.melody_embeddings(gt_melody[:,i])
            else:
                melody_embeddings = self.melody_embeddings(melody_note_prediction.argmax(dim=1))
            hx_melody, cx_melody = self.melody_lstm(melody_embeddings+chord_embeddings, (hx_melody, cx_melody))
            md = self.melody_prediction(hx_melody)
            melody_note_prediction = self.melody_note_prediction(md)
            note_outputs.append(melody_note_prediction)

        # stop when reaching max length
        for i in range(max_chords_length - 1):
            hx_chords, cx_chords = self.chords_lstm(chord_embeddings, (hx_chords, cx_chords))
            chord_prediction = self.chord_layers(hx_chords)
            if gt_chords is not None:
                chord_embeddings = self.chord_embeddings(gt_chords[:,i+1])
            else:
                chord_embeddings = self.chord_embeddings(chord_prediction.argmax(dim=1))
            chord_outputs.append(chord_prediction)

            for j in range(MELODY_DISCRETIZATION_LENGTH):
                if gt_melody is not None:
                    melody_embeddings = self.melody_embeddings(gt_melody[:, (i+1)*MELODY_DISCRETIZATION_LENGTH+j-1])
                else:
                    melody_embeddings = self.melody_embeddings(melody_note_prediction.argmax(dim=1))
                hx_melody, cx_melody = self.melody_lstm(melody_embeddings + chord_embeddings, (hx_melody, cx_melody))
                md = self.melody_prediction(hx_melody)
                melody_note_prediction = self.melody_note_prediction(md)
                note_outputs.append(melody_note_prediction)

        chord_outputs = torch.stack(chord_outputs, dim=1)
        melody_notes = torch.stack(note_outputs, dim=1)
        return chord_outputs, melody_notes
