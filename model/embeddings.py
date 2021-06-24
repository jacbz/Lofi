import numpy as np
import torch
from transformers import BertTokenizer, BertModel
from torch.nn.utils.rnn import pad_sequence
import json
import os

device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertModel.from_pretrained("bert-base-uncased").to(device)


# make embeddings and save as np
def make_embeddings():
    dataset_folder = "../datasets/processed"
    dataset_files = os.listdir(dataset_folder)
    embeddings_file = "embeddings"
    embedding_lengths_file = "embedding_lengths.json"
    embeddings = []
    embedding_lengths = {}
    with torch.no_grad():
        for i, file in enumerate(dataset_files):
            with open(f"{dataset_folder}/{file}", 'r') as json_file:
                json_parsed = json.load(json_file)
                lyrics = json_parsed["lyrics"]
                embedding, length = make_embedding(lyrics)
                embeddings.append(embedding.cpu())
                embedding_lengths[file] = length
            print(i)
    embeddings = pad_sequence(embeddings, batch_first=True).numpy()
    np.save(embeddings_file, embeddings)
    with open(embedding_lengths_file, 'w') as outfile:
        json.dump(embedding_lengths, outfile)


def make_embedding(lyrics):
    encoded_input = tokenizer(lyrics, truncation=True, return_tensors='pt').to(device)
    output = model(**encoded_input)
    embedding = output.last_hidden_state[0]
    length = output.last_hidden_state.shape[1]
    return embedding, length
