from flask import Flask, request, jsonify
from lofi_generate import generate
from lyrics2lofi_predict import predict
import torch
device = "cpu"

from model.lofi_model import LofiModel
from model.lyrics2lofi_model import Lyrics2LofiModel


app = Flask(__name__)

lofi_checkpoint = "../checkpoints/lofimodel_epoch850_weights_0.5to0.1over300_melodydelay250.pth"
print("Loading lofi model...", end=" ")
lofi_model = LofiModel(device=device)
lofi_model.load_state_dict(torch.load(lofi_checkpoint, map_location=device))
print(f"Loaded {lofi_checkpoint}.")
lofi_model.to(device)
lofi_model.eval()

lyrics2lofi_checkpoint = "../checkpoints/model-2021-07-09-02-19-260epochs.pth"
print("Loading lyrics2lofi model...", end=" ")
lyrics2lofi_model = Lyrics2LofiModel(device=device)
lyrics2lofi_model.load_state_dict(torch.load(lyrics2lofi_checkpoint, map_location=device))
print(f"Loaded {lyrics2lofi_checkpoint}.")
lyrics2lofi_model.to(device)
lyrics2lofi_model.eval()


@app.route('/')
def home():
    return 'Server running'


@app.route('/generate', methods=['GET'])
def sample_new_track():
    json_output = generate(lofi_model)
    response = jsonify(json_output)
    response.headers.add('Access-Control-Allow-Origin', '*')

    return response


@app.route('/predict', methods=['GET'])
def lyrics_to_track():
    input = request.args.get('input')
    json_output = predict(lyrics2lofi_model, input)
    response = jsonify(json_output)
    response.headers.add('Access-Control-Allow-Origin', '*')

    return response
