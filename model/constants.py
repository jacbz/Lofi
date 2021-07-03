BATCH_SIZE = 128
NUM_EPOCHS = 10000

BERT_EMBEDDING_LENGTH = 768

MAX_CHORD_PROGRESSION_LENGTH = 50
# input: hidden_size;output: chords (0-8) - 0: rest, 1-7: chord numerals, 8: end
CHORD_PREDICTION_LENGTH = 9
CHORD_PROGRESSION_REST_TOKEN = 0
CHORD_PROGRESSION_END_TOKEN = 8

MELODY_DISCRETIZATION_LENGTH = 16
# 23 melody classes: 0 (rest), 1 (repeat), 2-8 (octave 1), 9-15 (octave 2), 16-22 (octave 3)
MELODY_PREDICTION_LENGTH = 23

MELODY_REST_TOKEN = 0
MELODY_REPEAT_TOKEN = 1
MELODY_FIRST_SCALE_DEGREE = 2

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