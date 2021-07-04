BATCH_SIZE = 128
NUM_EPOCHS = 10000

BERT_EMBEDDING_LENGTH = 768

MAX_LENGTH_IN_MEASURES = 50

# input: hidden_size;output: chords (0-8) - 0: rest, 1-7: chord numerals, 8: end
CHORD_PREDICTION_LENGTH = 9
CHORD_REST_TOKEN = 0
CHORD_END_TOKEN = 8

# discretize each measure into this many chords, must be a divisor of MELODY_DISCRETIZATION_LENGTH
CHORD_DISCRETIZATION_LENGTH = 1
# discretize each measure into this many notes
MELODY_DISCRETIZATION_LENGTH = 16
NOTES_PER_CHORD = MELODY_DISCRETIZATION_LENGTH // CHORD_DISCRETIZATION_LENGTH

# 22 melody classes: 0 (rest), 1-7 (octave 1), 8-14 (octave 2), 15-21 (octave 3)
MELODY_PREDICTION_LENGTH = 22

MELODY_REST_TOKEN = 0

KEY_TO_NUM = {
    "C": 0,
    "C#": 1,
    "Db": 1,
    "D": 2,
    "D#": 3,
    "Eb": 3,
    "E": 4,
    "E#": 5,
    "F": 5,
    "F#": 6,
    "Gb": 6,
    "G": 7,
    "G#": 8,
    "Ab": 8,
    "A": 9,
    "A#": 10,
    "Bb": 10,
    "B": 11
}