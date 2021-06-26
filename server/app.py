from flask import Flask, request, jsonify
import json

import torch
from torch.nn.utils.rnn import pack_padded_sequence
import argparse

from embeddings import make_embedding
from model import Model

import jsonpickle

file = "model.pth"
device = "cuda" if torch.cuda.is_available() else "cpu"

app = Flask(__name__)

@app.route('/')
def home():
    return 'Encoder running'


@app.route('/encode', methods=['GET'])
def query_records():
    input = request.args.get('input')
    
    model = Model()
    model.load_state_dict(torch.load(file, map_location=torch.device('cpu')))

    model.to(device)
    model.eval()

    embedding, length = make_embedding(input)

    input = pack_padded_sequence(embedding[None], torch.tensor([length]), batch_first=True, enforce_sorted=False)
    pred, _ = model(input)

    chords = pred.argmax(dim=2)[0].tolist()
    chords.append(8)
    chords = chords[:chords.index(8) - 1]  # cut off 8

    print(f"Chord progression: {' '.join(map(str, chords))}")

    json_output = jsonpickle.encode(chords)
    return jsonify(json_output)