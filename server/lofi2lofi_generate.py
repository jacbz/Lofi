from server.output import *

def generate(model):
    hash, (pred_chords, pred_notes, pred_bpm, pred_key, pred_mode, pred_valence, pred_energy) = model.generate()

    output = Output(hash, pred_chords, pred_notes, pred_bpm, pred_key, pred_mode, pred_valence, pred_energy)

    json = output.to_json(True)
    print(json)
    return json
