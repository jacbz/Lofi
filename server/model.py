import torch
from torch import nn

device = "cuda" if torch.cuda.is_available() else "cpu"


class Model(nn.Module):
    def __init__(self, hidden_size=200, num_layers=1):
        super(Model, self).__init__()
        self.encoder = Encoder(hidden_size, num_layers)
        self.decoder = Decoder(hidden_size)
        self.mean_linear = nn.Linear(in_features=hidden_size, out_features=hidden_size)
        self.variance_linear = nn.Linear(in_features=hidden_size, out_features=hidden_size)

    def forward(self, input):
        # pred, (hidden states, cell states)
        _, (h, _) = self.encoder(input)

        # add two directions together
        temp = h[-1] + h[-2]

        # VAE
        mu = self.mean_linear(temp)
        logvar = self.variance_linear(temp)
        z = self.sample(mu, logvar)

        # make chord prediction
        chords_pred = self.decoder(z)

        # compute the Kullback–Leibler divergence between a Gaussian and an uniform Gaussian
        KL = 0.5 * torch.mean(mu ** 2 + logvar.exp() - logvar - 1, dim=[0, 1]) * 1e-2

        return chords_pred, KL

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
        # bert embedding length 768
        self.hidden_size = hidden_size
        self.rnn = nn.LSTM(input_size=768, hidden_size=hidden_size, num_layers=num_layers, bidirectional=True, batch_first=True)

    def forward(self, x):
        return self.rnn(x)


class Decoder(nn.Module):
    def __init__(self, hidden_size):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.cell = nn.LSTMCell(input_size=hidden_size * 1, hidden_size=hidden_size * 1)
        # input: hidden_size;output: chords (0-8) - 0: rest, 1-7: chord numerals, 8: end
        self.chord_linear = nn.Sequential(
            nn.Linear(in_features=hidden_size, out_features=hidden_size),
            nn.ReLU(),
            nn.Linear(in_features=hidden_size, out_features=9)
        )

    def forward(self, x):
        hx = torch.randn(x.shape[0], self.hidden_size * 1, device=device) # (batch, hidden_size)
        cx = torch.randn(x.shape[0], self.hidden_size * 1, device=device)
        hx, cx = self.cell(x, (hx, cx))
        chord_prediction = self.chord_linear(hx)
        output = [chord_prediction]

        # stop when reaching max length
        max_chord_progression_length = 325
        for i in range(max_chord_progression_length):
            hx, cx = self.cell(hx, (hx, cx))
            chord_prediction = self.chord_linear(hx)
            output.append(chord_prediction)
            # break when all have predicted an 8

        output = torch.stack(output, dim=1)
        preds = output.argmax(dim=2)
        preds[:,-1] = 8 # force last prediction to be 8
        return output