from flask import Flask, jsonify

from lofi_generate import generate

app = Flask(__name__)


@app.route('/')
def home():
    return 'Server running'


@app.route('/generate', methods=['GET'])
def query_records():
    json_output = generate()
    response = jsonify(json_output)
    response.headers.add('Access-Control-Allow-Origin', '*')

    return response