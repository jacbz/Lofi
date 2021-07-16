import collections

from model.constants import *


def process_sample(json_file):
    # between 0-11
    key = KEY_TO_NUM[json_file["metadata"]["key"]]

    mode_string = json_file["metadata"]["mode"]
    # if no mode_string, we can assume that it is major
    # between 0-6
    mode = int(mode_string) - 1 if mode_string is not None else 0

    energy = json_file["audio_features"]["energy"]
    valence = json_file["audio_features"]["valence"]

    bpm = int(json_file["audio_features"]["tempo"])
    # normalize tempo into range 70-100
    bpm = min(100.0, max(70.0, ((bpm - 70) * (3 / 13)) + 70))
    # normalize tempo into [0, 1], cutting off at 70 and 100
    tempo = (bpm / 30) - (7 / 3)

    json_chords = json_file["tracks"]["chord"]
    json_notes = json_file["tracks"]["melody"]

    octaves = [int(note["octave"]) for note in json_notes if not note["isRest"]]
    min_octave = min(octaves)
    max_octave = max(octaves)

    # find the NUMBER_OF_MELODY_OCTAVES octaves which the most notes, which we keep
    octave_occurrences = collections.Counter(octaves)
    octave_boundary_lower = min_octave
    count = 0
    for octave in range(min_octave, max_octave):
        curr_count = sum(octave_occurrences[o] for o in range(octave, octave + NUMBER_OF_MELODY_OCTAVES))
        if curr_count > count:
            count = curr_count
            octave_boundary_lower = octave

    beats_per_measure = int(json_file["metadata"]["beats_in_measure"])
    duration_in_beats = max(
        [chord["event_off"] for chord in json_chords] + [note["event_off"] for note in json_notes])
    num_measures = int(math.ceil(duration_in_beats / beats_per_measure))
    num_chords = num_measures * CHORD_DISCRETIZATION_LENGTH

    chords_list, note_list, num_chords = discretize_sample(json_chords, json_notes, octave_boundary_lower,
                                                           num_chords, num_measures * beats_per_measure)

    # pad chord and melodies to max measure length
    chords_list.append(CHORD_END_TOKEN)
    chords_list += [CHORD_END_TOKEN] * ((MAX_CHORD_LENGTH + 1) - len(chords_list))
    note_list += [MELODY_REST_TOKEN] * ((MAX_CHORD_LENGTH + 1) * NOTES_PER_CHORD - len(note_list))

    return {
        "key": key,
        "mode": mode,
        "chords": chords_list,
        "num_chords": num_chords,
        "melody_notes": note_list,
        "tempo": tempo,
        "energy": energy,
        "valence": valence
    }


# discretizes a sample into notes and chords
def discretize_sample(json_chords, json_notes, octave_boundary_lower, num_chords, max_event_off):
    chords = list(filter(lambda chord: not chord["isRest"], json_chords))
    notes = list(filter(lambda note: not note["isRest"], json_notes))

    chord_list = [CHORD_REST_TOKEN] * num_chords
    note_list = [MELODY_REST_TOKEN] * num_chords * NOTES_PER_CHORD

    for chord in reversed(chords):
        scale_degree = int(chord["sd"])
        relative_start = chord["event_on"] / max_event_off
        relative_end = chord["event_off"] / max_event_off
        i = round(relative_start * len(chord_list))
        j = round(relative_end * len(chord_list))
        for n in range(i, j):
            chord_list[n] = scale_degree

    for note in reversed(notes):
        scale_degree = int(note["scale_degree"].replace("s", "").replace("f", ""))
        octave = int(note["octave"])
        octave = octave - octave_boundary_lower
        octave = min(NUMBER_OF_MELODY_OCTAVES - 1, max(0, octave))
        scale_degree = octave * 7 + scale_degree

        relative_start = note["event_on"] / max_event_off
        relative_end = note["event_off"] / max_event_off
        i = round(relative_start * len(note_list))
        j = round(relative_end * len(note_list))
        for n in range(i, j):
            note_list[n] = scale_degree

    # delete empty chords with no melodies
    for i, chord in reversed(list(enumerate(chord_list))):
        if chord == CHORD_REST_TOKEN and all([note == MELODY_REST_TOKEN for note in
                                              note_list[i * NOTES_PER_CHORD:(i + 1) * NOTES_PER_CHORD]]):
            del chord_list[i]
            del note_list[i * NOTES_PER_CHORD:(i + 1) * NOTES_PER_CHORD]
            num_chords -= 1

    # trim to max chord length
    chord_list = chord_list[:MAX_CHORD_LENGTH]
    note_list = note_list[:MAX_CHORD_LENGTH * NOTES_PER_CHORD]

    return chord_list, note_list, min(MAX_CHORD_LENGTH, num_chords)
