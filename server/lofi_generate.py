import jsonpickle

from server.output import *
from model.constants import *


def generate(model):
    hash, (pred_chords, pred_notes, pred_bpm, pred_key, pred_mode, pred_valence, pred_energy) = model.generate()

    chords = pred_chords.argmax(dim=2)[0].tolist()
    notes = pred_notes.argmax(dim=2)[0].cpu().numpy()

    chords.append(CHORD_END_TOKEN)
    cut_off_point = chords.index(CHORD_END_TOKEN)
    chords = chords[:cut_off_point]  # cut off end token
    notes = notes[:cut_off_point * NOTES_PER_CHORD]

    title = hash
    key = pred_key.argmax().item() + 1
    mode = pred_mode.argmax().item() + 1
    bpm = round(pred_bpm.item() * 30 + 70)
    energy = round(pred_energy.item(), 3)
    valence = round(pred_valence.item(), 3)
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
