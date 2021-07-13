from model.constants import *
import jsonpickle

class Output:
    def __init__(self, title, pred_chords, pred_notes, pred_bpm, pred_key, pred_mode, pred_valence, pred_energy):
        chords = pred_chords.argmax(dim=2)[0].tolist()
        notes = pred_notes.argmax(dim=2)[0].cpu().numpy()

        chords.append(CHORD_END_TOKEN)
        cut_off_point = chords.index(CHORD_END_TOKEN)
        chords = chords[:cut_off_point]  # cut off end token
        notes = notes[:cut_off_point * NOTES_PER_CHORD]

        self.title = title
        self.key = pred_key.argmax().item() + 1
        self.mode = pred_mode.argmax().item() + 1
        self.bpm = round(pred_bpm.item() * 30 + 70)
        self.energy = round(pred_energy.item(), 3)
        self.valence = round(pred_valence.item(), 3)
        self.chords = chords
        melodies = notes.reshape(-1, NOTES_PER_CHORD)
        self.melodies = [x.tolist() for x in [*melodies]]

    def to_json(self, pretty=False):
        json = jsonpickle.encode(self, unpicklable=False)

        if pretty:
            json = json.replace(", \"", ",\n  \"") \
                .replace("{", "{\n  ") \
                .replace("}", "\n}") \
                .replace("[[", "[\n    [") \
                .replace("]]", "]\n  ]").replace("], [", "],\n    [")
        return json