import json

import torch
from torch.utils.data import Dataset

from model.dataset import *


class Lofi2LofiDataset(Dataset):
    def __init__(self, dataset_folder, files):
        super(Lofi2LofiDataset, self).__init__()
        self.samples = []

        for file in files:
            with open(f"{dataset_folder}/{file}") as sample_file_json:
                json_loaded = json.load(sample_file_json)
                sample = process_sample(json_loaded)
                self.samples.append(sample)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]

        return {
            "key": sample["key"],
            "mode": sample["mode"],
            "chords": torch.tensor(sample["chords"]),
            "num_chords": sample["num_chords"],
            "melody_notes": torch.tensor(sample["melody_notes"]),
            "tempo": sample["tempo"],
            "energy": sample["energy"],
            "valence": sample["valence"]
        }
