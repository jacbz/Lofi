from server.output import *


def generate(model):
    hash, (pred_chords, pred_notes, pred_tempo, pred_key, pred_mode, pred_valence, pred_energy) = model.generate()

    output = Output(hash, pred_chords, pred_notes, pred_tempo, pred_key, pred_mode, pred_valence, pred_energy)

    # try again if all chords or notes are zero (very rare)
    if all(c == 0 for c in output.chords) or all(n == 0 for arr in output.melodies for n in arr):
        print("All chords or melodies were zero, trying again...")
        return generate(model)

    json = output.to_json(True)
    print(json)
    return json


def decode(model, mu):
    hash, (pred_chords, pred_notes, pred_tempo, pred_key, pred_mode, pred_valence, pred_energy) = model.decode(mu)

    output = Output(hash, pred_chords, pred_notes, pred_tempo, pred_key, pred_mode, pred_valence, pred_energy)

    json = output.to_json(True)
    print(json)
    return json
