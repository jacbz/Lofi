import random

import jsonpickle
import numpy as np
import torch
import json
import os
from matplotlib.pyplot import plot, show, legend
from torch import nn
from torch.nn.utils import clip_grad_norm
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pack_padded_sequence

from constants import *
from model import Model
from predict import Output

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")



class SongDataset(Dataset):
    def __init__(self, files):
        super(SongDataset, self).__init__()
        self.samples = []
        self.embedding_lengths = []
        self.max_chord_progression_length = 0
        self.chord_count_map = {i: 0 for i in range(CHORD_PREDICTION_LENGTH)}
        self.melody_note_count_map = {i: 0 for i in range(MELODY_PREDICTION_LENGTH)}

        with open(embedding_lengths_file) as embeddings_length_json:
            embedding_lengths_json = json.load(embeddings_length_json)
            for file in files:
                with open(f"{dataset_folder}/{file}") as sample_file_json:
                    json_loaded = json.load(sample_file_json)

                    self.embedding_lengths.append(embedding_lengths_json[file])
                    sample = self.process_sample(json_loaded)
                    self.samples.append(sample)

        self.embeddings = np.load(f"{embeddings_file}.npy", mmap_mode="r")

    def process_sample(self, json_file):
        self.max_chord_progression_length = max(self.max_chord_progression_length, len(json_file["tracks"]["chord"]))

        # between 0-11
        key = KEY_TO_NUM[json_file["metadata"]["key"]]

        mode_string = json_file["metadata"]["mode"]
        # if no mode_string, we can assume that it is major
        # between 0-6
        mode = int(mode_string) - 1 if mode_string is not None else 0

        energy = json_file["audio_features"]["energy"]
        valence = json_file["audio_features"]["valence"]


        bpm = int(json_file["audio_features"]["tempo"])
        # normalize tempo into range 70-100
        bpm = min(100.0, max(70.0, ((bpm-70) * (3/13)) + 70))
        # normalize tempo into [0, 1], cutting off at 70 and 100
        bpm = (bpm/30) - (7/3)

        notes = json_file["tracks"]["melody"]

        octaves = [int(note["octave"]) for note in notes]
        min_octave = min(octaves)
        max_octave = max(octaves)
        octave_range = max_octave - min_octave

        # test if ranges [a1, a2] and [b1, b2] overlap
        def overlap(a1, a2, b1, b2):
            return max(a2, b2) - min(a1, b1) < (a2 - a1) + (b2 - b1)

        # returns a discretized note list of length MELODY_DISCRETIZATION_LENGTH for a chord
        def discretize_melodies(chord):
            chord_start = chord["event_on"]
            chord_end = chord["event_off"]
            notes_during_chord = filter(lambda note:  overlap(note["event_on"], note["event_off"], chord_start, chord_end), notes)
            note_list = [MELODY_REST_TOKEN] * MELODY_DISCRETIZATION_LENGTH

            for note in notes_during_chord:
                scale_degree = MELODY_REST_TOKEN if note["isRest"] else int(note["scale_degree"].replace("s", "").replace("f", ""))

                if scale_degree is not MELODY_REST_TOKEN:
                    # print(f"Min {min_octave}, max {max_octave}, range {octave_range}: ", end=" ")
                    octave = int(note["octave"])
                    # print(f"Octave {octave}->", end=" ")
                    octave = abs(abs(octave_range) - abs(octave)) if octave_range > 0 else 1

                    if octave > 2:
                        octave = 2
                    # print(f"{octave}, ", end=" ")

                    # offset
                    # print(f"sd {scale_degree}->", end=" ")
                    scale_degree = octave * 7 + scale_degree + MELODY_FIRST_SCALE_DEGREE - 1
                    # print(f"{scale_degree} ")

                relative_start = (max(chord_start, note["event_on"]) - chord_start) / (chord_end - chord_start)
                relative_end = (min(chord_end, note["event_off"]) - chord_start) / (chord_end - chord_start)
                i = round(relative_start * MELODY_DISCRETIZATION_LENGTH)
                j = round(relative_end * MELODY_DISCRETIZATION_LENGTH)
                for n in range(i, j):
                    note_list[n] = scale_degree if n == i or scale_degree == MELODY_REST_TOKEN else MELODY_REPEAT_TOKEN
            return note_list

        def map_chords(chord):
            if chord["sd"] == "rest":
                # map "rest" to 0
                return CHORD_PROGRESSION_REST_TOKEN
            return int(chord["sd"])

        json_chords = json_file["tracks"]["chord"][:MAX_CHORD_PROGRESSION_LENGTH-1]
        melody_note_list = []
        for note_list in list(map(discretize_melodies, json_chords)):
            melody_note_list.extend(note_list)

        for note in melody_note_list:
            self.melody_note_count_map[note] += 1
        melody_note_list += [0] * MELODY_DISCRETIZATION_LENGTH * (MAX_CHORD_PROGRESSION_LENGTH - len(melody_note_list)//MELODY_DISCRETIZATION_LENGTH)

        chords = list(map(map_chords, json_chords))

        output = Output(json_file["metadata"]["title"], key + 1, mode + 1, bpm, energy, valence, chords, [x.tolist() for x in [*np.array(melody_note_list[:len(chords) * MELODY_DISCRETIZATION_LENGTH]).reshape(-1, 16)]])
        output_json = jsonpickle.encode(output, unpicklable=False)\
        .replace(", \"", ",\n  \"")\
        .replace("{", "{\n  ")\
        .replace("}","\n}")\
        .replace("[[", "[\n    [")\
        .replace("]]","]\n  ]").replace("], [", "],\n    [")

        chords.append(CHORD_PROGRESSION_END_TOKEN)  # END token
        chords_length = len(chords)
        chords.append(CHORD_PROGRESSION_REST_TOKEN)

        # create chord count map
        for chord in chords:
            self.chord_count_map[chord] += 1

        # pad chord to max progression length
        chords += [CHORD_PROGRESSION_END_TOKEN] * (MAX_CHORD_PROGRESSION_LENGTH + 1 - len(chords))

        return {
                    "key": key,
                    "mode": mode,
                    "chords": chords,
                    "chords_length": chords_length,
                    "melody_notes": melody_note_list,
                    "bpm": bpm,
                    "energy": energy,
                    "valence": valence
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
                    "key": sample["key"],
                    "mode": sample["mode"],
                    "chords": torch.tensor(sample["chords"]),
                    "chords_length": sample["chords_length"],
                    "melody_notes": torch.tensor(sample["melody_notes"]),
                    "bpm": sample["bpm"],
                    "energy": sample["energy"],
                    "valence": sample["valence"]
                }



if __name__ == '__main__':
    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)

    dataset_folder = "dataset/processed-lyrics-spotify"
    dataset_files = os.listdir(dataset_folder)
    embeddings_file = "embeddings"
    embedding_lengths_file = "embedding_lengths.json"

    dataset = SongDataset(dataset_files)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # assign weights based on chord counts in dataset
    chord_weights = torch.tensor(list(dataset.chord_count_map.values()), dtype=torch.float32, device=device)
    chord_weights = chord_weights / chord_weights.sum()
    chord_weights = 1.0 / chord_weights
    chord_weights = chord_weights / chord_weights.sum()

    note_weights = torch.tensor(list(dataset.melody_note_count_map.values()), dtype=torch.float32, device=device)
    note_weights = note_weights / note_weights.sum()
    note_weights = 1.0 / note_weights
    note_weights = note_weights / note_weights.sum()

    chord_loss = nn.CrossEntropyLoss(reduction='none', weight=chord_weights)
    melody_loss_notes = nn.CrossEntropyLoss(reduction='none', weight=note_weights)
    mae = nn.L1Loss(reduction='mean')
    model = Model().to(device)
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    epochs = []
    losses_chords = []
    losses_melodies = []
    losses_kl = []
    # torch.autograd.set_detect_anomaly(True)
    for epoch in range(NUM_EPOCHS):
        print(f"== Epoch {epoch} ==")
        epoch_losses_chords = []
        epoch_losses_melodies = []
        epoch_losses_kl = []
        tp_chords = []
        tp_melodies = []

        for batch, data in enumerate(dataloader):
            embeddings = data["embedding"].to(device)
            embedding_lengths = data["embedding_length"]
            chords_length = data["chords_length"].to(device)
            max_chord_length = chords_length.max()
            chords = data["chords"].to(device)[:,:max_chord_length]
            melody_notes = data["melody_notes"].to(device)[:,:max_chord_length*MELODY_DISCRETIZATION_LENGTH]
            bpm_gt = data["bpm"].to(device)
            valence_gt = data["valence"].to(device)
            energy_gt = data["energy"].to(device)

            input = pack_padded_sequence(embeddings, embedding_lengths, batch_first=True, enforce_sorted=False)

            (pred_chords, pred_notes), kl, bpm, valence, energy = model(input, chords_length.max(), chords, melody_notes)
            loss_chords = chord_loss(pred_chords.permute(0, 2, 1), chords)
            loss_melody_notes = melody_loss_notes(pred_notes.permute(0, 2, 1), melody_notes)

            # compute mask
            arrange = torch.arange(max_chord_length, device=device).repeat((embeddings.shape[0], 1)).permute(0, 1)
            lengths_stacked = chords_length.repeat((max_chord_length, 1)).permute(1, 0)
            mask_chord = (arrange <= lengths_stacked)

            arrange = torch.arange(max_chord_length*MELODY_DISCRETIZATION_LENGTH, device=device).repeat((embeddings.shape[0], 1)).permute(0, 1)
            lengths_stacked = (chords_length*MELODY_DISCRETIZATION_LENGTH).repeat((max_chord_length*MELODY_DISCRETIZATION_LENGTH, 1)).permute(1, 0)
            mask_melody = (arrange <= lengths_stacked)

            loss_chords = torch.masked_select(loss_chords, mask_chord).mean()
            loss_melody = torch.masked_select(loss_melody_notes, mask_melody).mean()
            loss_melody *= 10
            kl_loss = kl * 1e-2
            loss_bpm = mae(bpm[:, 0], bpm_gt) / 5
            loss_valence = mae(valence[:, 0], valence_gt) / 5
            loss_energy = mae(energy[:,0], energy_gt) / 5
            loss = loss_chords + kl_loss + loss_melody + loss_bpm + loss_energy + loss_valence
            optimizer.zero_grad()
            loss.backward()

            optimizer.step()
            loss = loss.item()
            tp_chords.extend(torch.masked_select(pred_chords.argmax(dim=2) == chords, mask_chord).tolist())
            tp_melodies.extend(torch.masked_select(pred_notes.argmax(dim=2) == melody_notes, mask_melody).tolist())

            epoch_losses_chords.append(loss_chords)
            epoch_losses_melodies.append(loss_melody)
            epoch_losses_kl.append(kl_loss)
            print(f"\tBatch {batch}: Loss {loss} (C: {loss_chords} + KL: {kl_loss} + M: {loss_melody} + B: {loss_bpm} V: {loss_valence} + E: {loss_energy}), batch chord acc. cum. {sum(tp_chords) / len(tp_chords)}, batch melody acc. cum. {sum(tp_melodies) / len(tp_melodies)}")

        loss_chord = sum(epoch_losses_chords) / len(epoch_losses_chords)
        loss_melody = sum(epoch_losses_melodies) / len(epoch_losses_melodies)
        loss_kl = sum(epoch_losses_kl) / len(epoch_losses_kl)
        print(f"Epoch chord loss: {loss_chord}, melody loss: {loss_melody}, KL: {loss_kl}, chord accuracy: {sum(tp_chords) / len(tp_chords)}")
        torch.save(model.state_dict(), "model.pth")

        epochs.append(epoch)
        losses_chords.append(loss_chord)
        losses_melodies.append(loss_melody)
        losses_kl.append(loss_kl)
        plot(epochs, losses_chords, label='Chord Loss')
        plot(epochs, losses_melodies, label='Melody Loss')
        # plot(epochs, losses_kl, label='KL Divergence')
        legend()
        show()
