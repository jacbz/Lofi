import json

import numpy as np
from flask import Flask, request, jsonify
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
import torch

from server.lofi2lofi_generate import generate, decode
from server.lyrics2lofi_predict import predict
from model.lofi2lofi_model import Lofi2LofiModel
from model.lyrics2lofi_model import Lyrics2LofiModel

device = "cpu"
app = Flask(__name__)
limiter = Limiter(
    app,
    key_func=get_remote_address,
    default_limits=["30 per minute"]
)

lofi2lofi_checkpoint = "checkpoints/lofi2lofi.pth"
print("Loading lofi model...", end=" ")
lofi2lofi_model = Lofi2LofiModel(device=device)
lofi2lofi_model.load_state_dict(torch.load(lofi2lofi_checkpoint, map_location=device))
print(f"Loaded {lofi2lofi_checkpoint}.")
lofi2lofi_model.to(device)
lofi2lofi_model.eval()

# lyrics2lofi_checkpoint = "checkpoints/lyrics2lofi.pth"
# print("Loading lyrics2lofi model...", end=" ")
# lyrics2lofi_model = Lyrics2LofiModel(device=device)
# lyrics2lofi_model.load_state_dict(torch.load(lyrics2lofi_checkpoint, map_location=device))
# print(f"Loaded {lyrics2lofi_checkpoint}.")
# lyrics2lofi_model.to(device)
# lyrics2lofi_model.eval()


@app.route('/')
def home():
    return 'Server running'


@app.route('/generate', methods=['GET'])
def generate():
    json_output = generate(lofi2lofi_model)
    response = jsonify(json_output)
    response.headers.add('Access-Control-Allow-Origin', '*')

    return response

@app.route('/decode', methods=['GET'])
def decode_input():
    input = request.args.get('input')
    number_list = json.loads(input)
    json_output = decode(lofi2lofi_model, torch.tensor([number_list]).float())
    response = jsonify(json_output)
    response.headers.add('Access-Control-Allow-Origin', '*')

    return response

# @app.route('/predict', methods=['GET'])
# def lyrics_to_track():
#     input = request.args.get('input')
#     json_output = predict(lyrics2lofi_model, input)
#     response = jsonify(json_output)
#     response.headers.add('Access-Control-Allow-Origin', '*')
#
#     return response
