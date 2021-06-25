import numpy as np
import torch
import json
import os
from matplotlib.pyplot import plot, show
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pack_padded_sequence
from model import Model

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

MAX_CHORD_PROGRESSION_LENGTH = 325

class SongDataset(Dataset):
    def __init__(self, files):
        super(SongDataset, self).__init__()
        self.samples = []
        self.embedding_lengths = []
        self.max_chord_progression_length = 0
        self.chord_count_map = {}

        with open(embedding_lengths_file) as embeddings_length_json:
            embedding_lengths_json = json.load(embeddings_length_json)
            for file in files:
                with open(f"{dataset_folder}/{file}") as sample_file_json:
                    json_loaded = json.load(sample_file_json)
                    self.embedding_lengths.append(embedding_lengths_json[file])
                    sample = self.process_sample(json_loaded)
                    self.samples.append(sample)

        print(f"Max chord length: {self.max_chord_progression_length}")
        self.embeddings = np.load(f"{embeddings_file}.npy", mmap_mode="r")

    def process_sample(self, json_file):
        self.max_chord_progression_length = max(self.max_chord_progression_length, len(json_file["tracks"]["chord"]))

        bpm = [int(json_file["metadata"]["BPM"])]

        def map_chords(chord):
            if chord["sd"] == "rest":
                # map "rest" to 0
                return 0
            return int(chord["sd"])

        chords = list(map(map_chords, json_file["tracks"]["chord"]))
        chords.append(8)  # END token
        chords_length = len(chords)

        # create chord count map
        for chord in chords:
            if chord not in self.chord_count_map.keys():
                self.chord_count_map[chord] = 1
            else:
                self.chord_count_map[chord] += 1

        # pad chord to max progression length
        chords += [8] * (MAX_CHORD_PROGRESSION_LENGTH + 1 - len(chords))

        return {
                    "bpm": bpm,
                    "chords": chords,
                    "chords_length": chords_length,
                    #"melody": torch.tensor(list(map(lambda melody: int(melody["sd"]) if melody["sd"] != "rest" else 0, sample["tracks"]["melody"]))[:4])
                }

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        embedding = np.copy(self.embeddings[index])
        embedding_length = self.embedding_lengths[index]

        return {
                    "embedding": embedding,
                    "embedding_length": embedding_length,
                    "bpm": torch.tensor(sample["bpm"]),
                    "chords": torch.tensor(sample["chords"]),
                    "chords_length": sample["chords_length"],
                }



if __name__ == '__main__':
    dataset_folder = "dataset/processed"
    dataset_files = os.listdir(dataset_folder)
    embeddings_file = "embeddings"
    embedding_lengths_file = "embedding_lengths.json"

    dataset = SongDataset(dataset_files)
    batch_size = 128
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # assign weights based on chord counts in dataset
    chord_counts = []
    for i in range(9):
        chord_counts.append(dataset.chord_count_map[i])
    weights = torch.tensor(chord_counts, dtype=torch.float32, device=device)
    weights = weights / weights.sum()
    weights = 1.0 / weights
    weights = weights / weights.sum()

    loss_fn = nn.CrossEntropyLoss(reduction='none', weight=weights)
    model = Model().to(device)
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    num_epochs = 10000
    epochs = []
    losses = []

    for epoch in range(num_epochs):
        print(f"== Epoch {epoch} ==")
        epoch_losses = []
        true_positives = []
        for batch, data in enumerate(dataloader):
            embeddings = data["embedding"].to(device)
            embedding_lengths = data["embedding_length"]
            bpm = data["bpm"].to(device)
            chords = data["chords"].to(device)
            chords_length = data["chords_length"].to(device)

            input = pack_padded_sequence(embeddings, embedding_lengths, batch_first=True, enforce_sorted=False)

            pred, KL = model(input)
            loss = loss_fn(pred.permute(0, 2, 1), chords)

            # compute mask
            arrange = torch.arange(MAX_CHORD_PROGRESSION_LENGTH + 1, device=device).repeat((embeddings.shape[0], 1)).permute(0, 1)
            lengths_stacked = chords_length.repeat((MAX_CHORD_PROGRESSION_LENGTH + 1, 1)).permute(1, 0)
            mask = (arrange <= lengths_stacked)

            loss = torch.masked_select(loss, mask).mean() + KL

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss = loss.item()
            true_positives.extend(torch.masked_select(pred.argmax(dim=2) == chords, mask).tolist())
            epoch_losses.append(loss)
            print(f"\tBatch {batch}: Loss {loss}, batch accuracy cumulative {sum(true_positives) / len(true_positives)}")

        loss = sum(epoch_losses) / len(epoch_losses)
        print(f"Epoch loss: {loss}, epoch accuracy: {sum(true_positives) / len(true_positives)}")
        torch.save(model.state_dict(), "model.pth")

        epochs.append(epoch)
        losses.append(loss)
        plot(epochs, losses)
        show()
