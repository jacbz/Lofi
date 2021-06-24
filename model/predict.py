import torch
from torch.nn.utils.rnn import pack_padded_sequence
import argparse

from embeddings import make_embedding
from model import Model

device = "cuda" if torch.cuda.is_available() else "cpu"

parser = argparse.ArgumentParser(description='Generate some chords.')
parser.add_argument('input', type=str)
args = parser.parse_args()

if __name__ == '__main__':
    model = Model()
    model.load_state_dict(torch.load("model.pth"))
    model.to(device)

    model.eval()

    input = args.input
    embedding, length = make_embedding(input)

    input = pack_padded_sequence(embedding[None], torch.tensor([length]), batch_first=True, enforce_sorted=False)
    pred, _ = model(input)

    chords = pred.argmax(dim=2)[0].tolist()
    chords.append(8)
    chords = chords[:chords.index(8)-1]  # cut off 8

    print(f"Chord progression: {' '.join(map(str, chords))}")