import jsonpickle
import torch
from torch.nn.utils.rnn import pack_padded_sequence

from server.output import *
from model.constants import *
from model.embeddings import make_embedding

device = "cpu"


def predict(model, input):
    embedding, length = make_embedding(input, device)

    input = pack_padded_sequence(embedding[None], torch.tensor([length]), batch_first=True, enforce_sorted=False)
    pred_chords, pred_notes, pred_bpm, pred_key, pred_mode, pred_valence, pred_energy, kl = model(input, MAX_CHORD_LENGTH)

    chords = pred_chords.argmax(dim=2)[0].tolist()
    notes = pred_notes.argmax(dim=2)[0].cpu().numpy()

    chords.append(CHORD_END_TOKEN)
    cut_off_point = chords.index(CHORD_END_TOKEN)
    chords = chords[:cut_off_point]  # cut off end token
    notes = notes[:cut_off_point * NOTES_PER_CHORD]

    title = None
    key = pred_key.argmax().item() + 1
    mode = pred_mode.argmax().item() + 1
    bpm = round(pred_bpm.item() * 30 + 70)
    energy = pred_energy.item()
    valence = pred_valence.item()
    chords = chords
    melodies = notes.reshape(-1, NOTES_PER_CHORD)
    melodies = [x.tolist() for x in [*melodies]]

    output = Output(title, key, mode, bpm, energy, valence, chords, melodies)

    json = jsonpickle.encode(output, unpicklable=False)\
        .replace(", \"", ",\n  \"")\
        .replace("{", "{\n  ")\
        .replace("}","\n}")\
        .replace("[[", "[\n    [")\
        .replace("]]","]\n  ]").replace("], [", "],\n    [")
    print(json)
    return json
	