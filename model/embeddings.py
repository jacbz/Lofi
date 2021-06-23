import numpy as np
import torch
from transformers import BertTokenizer, BertModel
from torch.nn.utils.rnn import pad_sequence
import json
import os

dataset_folder = "../datasets/processed"
dataset_files = os.listdir(dataset_folder)
embeddings_file = "embeddings"
embedding_lengths_file = "embedding_lengths.json"

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

# make embeddings and save as np
def make_embeddings():
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertModel.from_pretrained("bert-base-uncased").to(device)
    embeddings = []
    embedding_lengths = {}
    with torch.no_grad():
        for i, file in enumerate(dataset_files):
            with open(f"{dataset_folder}/{file}", 'r') as json_file:
                json_parsed = json.load(json_file)
                lyrics = json_parsed["lyrics"]
                encoded_input = tokenizer(lyrics, truncation=True, return_tensors='pt').to(device)
                output = model(**encoded_input)
                embedding_lengths[file] = output.last_hidden_state.shape[1]
                embeddings.append(output.last_hidden_state[0].cpu())
            print(i)
    embeddings = pad_sequence(embeddings, batch_first=True).numpy()
    np.save(embeddings_file, embeddings)
    with open(embedding_lengths_file, 'w') as outfile:
        json.dump(embedding_lengths, outfile)
