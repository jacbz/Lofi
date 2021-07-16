import torch
from torch.nn.utils.rnn import pack_padded_sequence

from model.embeddings import make_embedding
from server.output import *


def predict(model, input):
    embedding, length = make_embedding(input, "cpu")

    input = pack_padded_sequence(embedding[None], torch.tensor([length]), batch_first=True, enforce_sorted=False)
    pred_chords, pred_notes, pred_tempo, pred_key, pred_mode, pred_valence, pred_energy, kl = model(input)

    output = Output(input, pred_chords, pred_notes, pred_tempo, pred_key, pred_mode, pred_valence, pred_energy)

    json = output.to_json(True)
    print(json)
    return json
