import json

import numpy as np
import torch
from torch.utils.data import Dataset

from model.dataset import *


class Lyrics2LofiDataset(Dataset):
    def __init__(self, dataset_folder, files, embeddings_file, embedding_lengths_file):
        super(Lyrics2LofiDataset, self).__init__()
        self.samples = []
        self.embedding_lengths = []

        with open(embedding_lengths_file) as embeddings_length_json:
            embedding_lengths_json = json.load(embeddings_length_json)
            for file in files:
                with open(f"{dataset_folder}/{file}") as sample_file_json:
                    json_loaded = json.load(sample_file_json)
                    self.embedding_lengths.append(embedding_lengths_json[file])
                    sample = process_sample(json_loaded)
                    self.samples.append(sample)

        self.embeddings = np.load(f"{embeddings_file}.npy", mmap_mode="r")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        embedding = np.copy(self.embeddings[index])
        embedding_length = self.embedding_lengths[index]

        return {
            "embedding": embedding,
            "embedding_length": embedding_length,
            "key": sample["key"],
            "mode": sample["mode"],
            "chords": torch.tensor(sample["chords"]),
            "num_chords": sample["num_chords"],
            "melody_notes": torch.tensor(sample["melody_notes"]),
            "tempo": sample["tempo"],
            "energy": sample["energy"],
            "valence": sample["valence"]
        }
