import math
import random

import jsonpickle
import numpy as np
import torch
import json
import os
import matplotlib.pyplot as plot
from sklearn.model_selection import KFold
from torch import nn
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler
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

        json_chords = json_file["tracks"]["chord"]
        json_notes = json_file["tracks"]["melody"]

        octaves = [int(note["octave"]) for note in json_notes]
        min_octave = min(octaves)
        max_octave = max(octaves)
        octave_range = max_octave - min_octave

        # validation if ranges [a1, a2] and [b1, b2] overlap
        def overlap(a1, a2, b1, b2):
            return max(a2, b2) - min(a1, b1) < (a2 - a1) + (b2 - b1)

        # discretizes a measure into
        # CHORD_DISCRETIZATION_LENGTH chords
        # MELODY_DISCRETIZATION_LENGTH notes
        def discretize(measure_start, measure_end):
            chords_during_measure = filter(lambda chord: overlap(chord["event_on"], chord["event_off"], measure_start, measure_end), json_chords)
            notes_during_measure = filter(lambda note:  overlap(note["event_on"], note["event_off"], measure_start, measure_end), json_notes)

            chord_list = [CHORD_REST_TOKEN] * CHORD_DISCRETIZATION_LENGTH
            note_list = [MELODY_REST_TOKEN] * MELODY_DISCRETIZATION_LENGTH
            isEmpty = True

            for chord in chords_during_measure:
                if chord["isRest"]:
                    continue
                scale_degree = int(chord["sd"])
                relative_start = (max(measure_start, chord["event_on"]) - measure_start) / (measure_end - measure_start)
                relative_end = (min(measure_end, chord["event_off"]) - measure_start) / (measure_end - measure_start)
                i = round(relative_start * CHORD_DISCRETIZATION_LENGTH)
                j = round(relative_end * CHORD_DISCRETIZATION_LENGTH)
                for n in range(i, j):
                    isEmpty = False
                    chord_list[n] = scale_degree

            for note in notes_during_measure:
                if note["isRest"]:
                    continue
                scale_degree = int(note["scale_degree"].replace("s", "").replace("f", ""))

                if scale_degree is not MELODY_REST_TOKEN:
                    octave = int(note["octave"])
                    octave = min(2, abs(abs(octave_range) - abs(octave)) if octave_range > 0 else 1)
                    scale_degree = octave * 7 + scale_degree

                relative_start = (max(measure_start, note["event_on"]) - measure_start) / (measure_end - measure_start)
                relative_end = (min(measure_end, note["event_off"]) - measure_start) / (measure_end - measure_start)
                i = round(relative_start * MELODY_DISCRETIZATION_LENGTH)
                j = round(relative_end * MELODY_DISCRETIZATION_LENGTH)
                for n in range(i, j):
                    isEmpty = False
                    note_list[n] = scale_degree

            return chord_list, note_list, isEmpty


        beats_per_measure = int(json_file["metadata"]["beats_in_measure"])
        duration = max([chord["event_off"] for chord in json_chords] + [note["event_off"] for note in json_notes])
        num_measures = min(MAX_LENGTH_IN_MEASURES, int(math.ceil(duration / beats_per_measure)))

        chords_list = []
        melody_note_list = []
        for measure in range(num_measures):
            measure_start = measure * beats_per_measure
            measure_end = (measure + 1) * beats_per_measure
            measure_chord_list, measure_note_list, isEmpty = discretize(measure_start, measure_end)
            if isEmpty:
                num_measures -= 1
            else:
                chords_list.extend(measure_chord_list)
                melody_note_list.extend(measure_note_list)

        output = Output(json_file["metadata"]["title"], key + 1, mode + 1, bpm, energy, valence, chords_list, [x.tolist() for x in [*np.array(melody_note_list[:len(chords_list) * MELODY_DISCRETIZATION_LENGTH]).reshape(-1, 16)]])
        output_json = jsonpickle.encode(output, unpicklable=False)\
        .replace(", \"", ",\n  \"")\
        .replace("{", "{\n  ")\
        .replace("}","\n}")\
        .replace("[[", "[\n    [")\
        .replace("]]","]\n  ]").replace("], [", "],\n    [")

        chords_list.append(CHORD_END_TOKEN)  # END token
        for chord in chords_list:
            self.chord_count_map[chord] += 1
        for note in melody_note_list:
            self.melody_note_count_map[note] += 1

        # pad chord and melodies to max measure length
        chords_list += [CHORD_END_TOKEN] * ((MAX_LENGTH_IN_MEASURES + 1) * CHORD_DISCRETIZATION_LENGTH - len(chords_list))
        melody_note_list += [MELODY_REST_TOKEN] * ((MAX_LENGTH_IN_MEASURES + 1) * MELODY_DISCRETIZATION_LENGTH - len(melody_note_list))

        return {
                    "key": key,
                    "mode": mode,
                    "chords": chords_list,
                    "num_measures": num_measures,
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
                    "num_measures": sample["num_measures"],
                    "melody_notes": torch.tensor(sample["melody_notes"]),
                    "bpm": sample["bpm"],
                    "energy": sample["energy"],
                    "valence": sample["valence"]
                }


# losses for one batch of data
def compute_loss(data):
    embeddings = data["embedding"].to(device)
    embedding_lengths = data["embedding_length"]
    num_measures = data["num_measures"].to(device)
    max_num_measures = num_measures.max()
    max_num_chords = max_num_measures * CHORD_DISCRETIZATION_LENGTH
    max_num_notes = max_num_measures * MELODY_DISCRETIZATION_LENGTH

    chords_gt = data["chords"].to(device)[:, :max_num_chords]
    notes_gt = data["melody_notes"].to(device)[:, :max_num_notes]
    bpm_gt = data["bpm"].to(device)
    valence_gt = data["valence"].to(device)
    energy_gt = data["energy"].to(device)

    input = pack_padded_sequence(embeddings, embedding_lengths, batch_first=True, enforce_sorted=False)

    pred_chords, pred_notes, pred_bpm, pred_valence, pred_energy, kl = model(input, max_num_measures.max(), chords_gt, notes_gt)
    loss_chords = chord_loss(pred_chords.permute(0, 2, 1), chords_gt)
    loss_melody_notes = melody_loss_notes(pred_notes.permute(0, 2, 1), notes_gt)

    # compute mask
    arange = torch.arange(max_num_chords, device=device).repeat((embeddings.shape[0], 1)).permute(0, 1)
    lengths_stacked = (num_measures * CHORD_DISCRETIZATION_LENGTH).repeat((max_num_chords, 1)).permute(1, 0)
    mask_chord = (arange <= lengths_stacked)

    arange = torch.arange(max_num_notes, device=device).repeat((embeddings.shape[0], 1)).permute(0, 1)
    lengths_stacked = (num_measures * MELODY_DISCRETIZATION_LENGTH).repeat((max_num_notes, 1)).permute(1, 0)
    mask_melody = (arange <= lengths_stacked)

    loss_chords = torch.masked_select(loss_chords, mask_chord).mean()
    loss_melody = torch.masked_select(loss_melody_notes, mask_melody).mean()
    loss_melody *= 10
    kl_loss = kl * 1e-2
    loss_bpm = mae(pred_bpm[:, 0], bpm_gt) / 5
    loss_valence = mae(pred_valence[:, 0], valence_gt) / 5
    loss_energy = mae(pred_energy[:, 0], energy_gt) / 5
    loss = loss_chords + kl_loss + loss_melody + loss_bpm + loss_energy + loss_valence
    tp_chords = torch.masked_select(pred_chords.argmax(dim=2) == chords_gt, mask_chord).tolist()
    tp_melodies = torch.masked_select(pred_notes.argmax(dim=2) == notes_gt, mask_melody).tolist()

    return loss, loss_chords, kl_loss, loss_melody, loss_bpm, loss_valence, loss_energy, tp_chords, tp_melodies

if __name__ == '__main__':
    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)

    dataset_folder = "dataset/processed-lyrics-spotify"
    dataset_files = os.listdir(dataset_folder)
    embeddings_file = "embeddings"
    embedding_lengths_file = "embedding_lengths.json"

    dataset = SongDataset(dataset_files)
    train_size = int(0.8 * len(dataset))
    validation_size = len(dataset) - train_size
    train_dataset, validation_dataset = torch.utils.data.random_split(dataset, [train_size, validation_size])
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    validation_dataloader = DataLoader(validation_dataset, batch_size=BATCH_SIZE, shuffle=False)

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
    # model.load_state_dict(torch.load("model-2021-07-04-15-57-370epochs.pth"))

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    epochs = []
    training_losses_chords = []
    training_losses_melodies = []
    training_losses_kl = []
    training_accuracies_chords = []
    training_accuracies_melodies = []

    validation_losses_chords = []
    validation_losses_melodies = []
    validation_losses_kl = []
    validation_accuracies_chords = []
    validation_accuracies_melodies = []

    for epoch in range(NUM_EPOCHS):
        print(f"== Epoch {epoch} ==")
        epoch_training_losses_chords = []
        epoch_training_losses_melodies = []
        epoch_training_losses_kl = []
        epoch_training_tp_chords = []
        epoch_training_tp_melodies = []

        epoch_validation_losses_chords = []
        epoch_validation_losses_melodies = []
        epoch_validation_losses_kl = []
        epoch_validation_tp_chords = []
        epoch_validation_tp_melodies = []

        model.train()
        for batch, data in enumerate(train_dataloader):
            loss, loss_chords, kl_loss, epoch_training_loss_melody, loss_bpm, loss_valence, loss_energy, batch_tp_chords, batch_tp_melodies = compute_loss(data)

            epoch_training_losses_chords.append(loss_chords)
            epoch_training_losses_melodies.append(epoch_training_loss_melody)
            epoch_training_losses_kl.append(kl_loss)
            epoch_training_tp_chords.extend(batch_tp_chords)
            epoch_training_tp_melodies.extend(batch_tp_melodies)

            optimizer.zero_grad()
            loss.backward()

            optimizer.step()
            loss = loss.item()
            print(f"\tBatch {batch}: Loss {loss} (C: {loss_chords} + KL: {kl_loss} + M: {epoch_training_loss_melody} + B: {loss_bpm} V: {loss_valence} + E: {loss_energy}), batch chord acc. cum. {sum(epoch_training_tp_chords) / len(epoch_training_tp_chords)}, batch melody acc. cum. {sum(epoch_training_tp_melodies) / len(epoch_training_tp_melodies)}")

        model.eval()
        for batch, data in enumerate(validation_dataloader):
            with torch.no_grad():
                loss, loss_chords, kl_loss, epoch_training_loss_melody, loss_bpm, loss_valence, loss_energy, batch_tp_chords, batch_tp_melodies = compute_loss(data)

                epoch_validation_losses_chords.append(loss_chords)
                epoch_validation_losses_melodies.append(epoch_training_loss_melody)
                epoch_validation_losses_kl.append(kl_loss)
                epoch_validation_tp_chords.extend(batch_tp_chords)
                epoch_validation_tp_melodies.extend(batch_tp_melodies)

                print(
                    f"\tVALIDATION Batch {batch}: Loss {loss} (C: {loss_chords} + KL: {kl_loss} + M: {epoch_training_loss_melody} + B: {loss_bpm} V: {loss_valence} + E: {loss_energy}), batch chord acc. cum. {sum(epoch_validation_tp_chords) / len(epoch_validation_tp_chords)}, batch melody acc. cum. {sum(epoch_validation_tp_melodies) / len(epoch_validation_tp_melodies)}")

        epoch_training_loss_chord = sum(epoch_training_losses_chords) / len(epoch_training_losses_chords)
        epoch_training_loss_melody = sum(epoch_training_losses_melodies) / len(epoch_training_losses_melodies)
        epoch_training_loss_kl = sum(epoch_training_losses_kl) / len(epoch_training_losses_kl)
        epoch_training_chord_accuracy = (sum(epoch_training_tp_chords) / len(epoch_training_tp_chords)) * 100
        epoch_training_melody_accuracy = (sum(epoch_training_tp_melodies) / len(epoch_training_tp_melodies)) * 100

        epoch_validation_loss_chord = sum(epoch_validation_losses_chords) / len(epoch_validation_losses_chords)
        epoch_validation_loss_melody = sum(epoch_validation_losses_melodies) / len(epoch_validation_losses_melodies)
        epoch_validation_loss_kl = sum(epoch_validation_losses_kl) / len(epoch_validation_losses_kl)
        epoch_validation_chord_accuracy = (sum(epoch_validation_tp_chords) / len(epoch_validation_tp_chords)) * 100
        epoch_validation_melody_accuracy = (sum(epoch_validation_tp_melodies) / len(epoch_validation_tp_melodies)) * 100

        print(f"Epoch chord loss: {epoch_training_loss_chord}, melody loss: {epoch_training_loss_melody}, KL: {epoch_training_loss_kl}, "
              f"chord accuracy: {epoch_training_chord_accuracy}, melody accuracy: {epoch_training_melody_accuracy}")
        print(f"VALIDATION: epoch chord loss: {epoch_validation_loss_chord}, melody loss: {epoch_validation_loss_melody}, KL: {epoch_validation_loss_kl}, "
              f"chord accuracy: {epoch_validation_chord_accuracy}, melody accuracy: {epoch_validation_melody_accuracy}")
        torch.save(model.state_dict(), "model.pth")

        epochs.append(epoch)

        training_losses_chords.append(epoch_training_loss_chord)
        training_losses_melodies.append(epoch_training_loss_melody)
        training_losses_kl.append(epoch_training_loss_kl)
        training_accuracies_chords.append(epoch_training_chord_accuracy)
        training_accuracies_melodies.append(epoch_training_melody_accuracy)

        validation_losses_chords.append(epoch_validation_loss_chord)
        validation_losses_melodies.append(epoch_validation_loss_melody)
        validation_losses_kl.append(epoch_validation_loss_kl)
        validation_accuracies_chords.append(epoch_validation_chord_accuracy)
        validation_accuracies_melodies.append(epoch_validation_melody_accuracy)

        fig, axs = plot.subplots(2, 2)

        # Chords loss
        axs[0, 0].set_title('Chords loss')
        axs[0, 0].plot(epochs, training_losses_chords, label='Train', color='royalblue')
        axs[0, 0].plot(epochs, validation_losses_chords, label='Val', color='royalblue', linestyle='dotted')
        axs[0, 0].set_xlabel('Epochs')
        axs[0, 0].set_ylabel('Loss')
        axs[0, 0].set_ylim(bottom=0)
        axs[0, 0].legend()
        axs[0, 0].grid(True)

        # Chords accuracy
        axs[1, 0].set_title('Chords accuracy')
        axs[1, 0].plot(epochs, training_accuracies_chords, label='Train', color='darkorange')
        axs[1, 0].plot(epochs, validation_accuracies_chords, label='Val', color='darkorange', linestyle='dotted')
        axs[1, 0].set_xlabel('Epochs')
        axs[1, 0].set_ylabel('Accuracy (%)')
        axs[1, 0].set_ylim(bottom=0, top=100)
        axs[1, 0].legend()
        axs[1, 0].grid(True)

        # Melody loss
        axs[0, 1].set_title('Melody loss')
        axs[0, 1].plot(epochs, training_losses_melodies, label='Train', color='royalblue')
        axs[0, 1].plot(epochs, validation_losses_melodies, label='Val', color='royalblue', linestyle='dotted')
        axs[0, 1].set_xlabel('Epochs')
        axs[0, 1].set_ylabel('Loss')
        axs[0, 1].set_ylim(bottom=0)
        axs[0, 1].legend()
        axs[0, 1].grid(True)

        # Melody accuracy
        axs[1, 1].set_title('Melody accuracy')
        axs[1, 1].plot(epochs, training_accuracies_melodies, label='Train', color='darkorange')
        axs[1, 1].plot(epochs, validation_accuracies_melodies, label='Val', color='darkorange', linestyle='dotted')
        axs[1, 1].set_xlabel('Epochs')
        axs[1, 1].set_ylabel('Accuracy (%)')
        axs[1, 1].set_ylim(bottom=0, top=100)
        axs[1, 1].legend()
        axs[1, 1].grid(True)

        plot.tight_layout()
        plot.show()
