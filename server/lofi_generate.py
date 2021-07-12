import jsonpickle

from server.output import *
from model.constants import *


def generate(model):
    hash, (pred_chords, pred_notes) = model.generate()

    chords = pred_chords.argmax(dim=2)[0].tolist()
    notes = pred_notes.argmax(dim=2)[0].cpu().numpy()

    chords.append(CHORD_END_TOKEN)
    cut_off_point = chords.index(CHORD_END_TOKEN)
    print(cut_off_point)
    chords = chords[:cut_off_point]  # cut off end token
    notes = notes[:cut_off_point * NOTES_PER_CHORD]

    title = hash
    #key = pred_key.argmax().item() + 1
    key = 1
    #mode = pred_mode.argmax().item() + 1
    mode = 1
    #bpm = round(pred_bpm.item() * 30 + 70)
    bpm = 80
    #energy = pred_energy.item()
    energy = 0.5
    #valence = pred_valence.item()
    valence = 0.5
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
