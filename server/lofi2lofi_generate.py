import torch

from server.output import *


def decode(model, mu):
    hash, (pred_chords, pred_notes, pred_tempo, pred_key, pred_mode, pred_valence, pred_energy) = model.decode(mu)

    output = Output(hash, pred_chords, pred_notes, pred_tempo, pred_key, pred_mode, pred_valence, pred_energy)

    json = output.to_json(True)
    print(json)
    return json


def generate(model):
    mu = torch.randn(1, HIDDEN_SIZE)
    return decode(model, mu)
