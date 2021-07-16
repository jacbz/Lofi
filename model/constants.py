import math

BATCH_SIZE = 128
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.1
TRAIN_VALIDATION_SPLIT = 0.85

# whether to enable teacher forcing
# see Goyal, Anirudh et al.(2016): Professor forcing: a new algorithm for training recurrent networks
TEACHER_FORCE = False
# scheduled sampling parameters, when teacher forcing is active
# see Bengio, Samy et al. (2015): Scheduled Sampling for Sequence Prediction with Recurrent Neural Networks
SCHEDULED_SAMPLING_CONVERGENCE = 30
START_SCHEDULED_SAMPLING_RATE = 0.5
END_SCHEDULED_SAMPLING_RATE = 0.1
# number of epochs to wait before adding the melody loss
MELODY_EPOCH_DELAY = 0


# inverse sigmoid decay
def sampling_rate_at_epoch(epoch):
    if epoch < 0:
        return START_SCHEDULED_SAMPLING_RATE
    return (SCHEDULED_SAMPLING_CONVERGENCE / (
                SCHEDULED_SAMPLING_CONVERGENCE + math.exp(epoch / SCHEDULED_SAMPLING_CONVERGENCE))) * (
                       START_SCHEDULED_SAMPLING_RATE - END_SCHEDULED_SAMPLING_RATE) + END_SCHEDULED_SAMPLING_RATE


HIDDEN_SIZE = 100
HIDDEN_SIZE2 = 32
NUM_LAYERS = 1

BERT_EMBEDDING_LENGTH = 768
MAX_CHORD_LENGTH = 50

# input: hidden_size;output: chords (0-8) - 0: rest, 1-7: chord numerals, 8: end
CHORD_PREDICTION_LENGTH = 9
CHORD_REST_TOKEN = 0
CHORD_END_TOKEN = 8

# discretize each measure into this many chords, must be a divisor of MELODY_DISCRETIZATION_LENGTH
CHORD_DISCRETIZATION_LENGTH = 1
# discretize each measure into this many notes
MELODY_DISCRETIZATION_LENGTH = 8
NOTES_PER_CHORD = MELODY_DISCRETIZATION_LENGTH // CHORD_DISCRETIZATION_LENGTH

# 15 melody classes: 0 (rest), 1-7 (octave 1), 8-14 (octave 2)
NUMBER_OF_MELODY_OCTAVES = 2
MELODY_PREDICTION_LENGTH = 1 + NUMBER_OF_MELODY_OCTAVES * 7
MELODY_REST_TOKEN = 0

NUMBER_OF_KEYS = 12
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

NUMBER_OF_MODES = 7
