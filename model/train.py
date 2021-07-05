import collections
import math
import random

import jsonpickle
import numpy as np
import torch
import json
import os
import matplotlib.pyplot as plot
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pack_padded_sequence

from constants import *
from dataset import SongDataset
from model import Model

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")


# losses for one batch of data
def compute_loss(data):
    embeddings = data["embedding"].to(device)
    embedding_lengths = data["embedding_length"]
    num_chords = data["num_chords"].to(device)
    num_notes = num_chords * NOTES_PER_CHORD
    max_num_chords = num_chords.max()
    max_num_notes = max_num_chords * NOTES_PER_CHORD

    chords_gt = data["chords"].to(device)[:, :max_num_chords]
    notes_gt = data["melody_notes"].to(device)[:, :max_num_notes]
    bpm_gt = data["bpm"].to(device)
    valence_gt = data["valence"].to(device)
    energy_gt = data["energy"].to(device)

    input = pack_padded_sequence(embeddings, embedding_lengths, batch_first=True, enforce_sorted=False)

    # run model
    pred_chords, pred_notes, pred_bpm, pred_valence, pred_energy, kl = model(input, max_num_chords, chords_gt, notes_gt)

    loss_chords = chord_loss(pred_chords.permute(0, 2, 1), chords_gt)
    loss_melody_notes = melody_loss_notes(pred_notes.permute(0, 2, 1), notes_gt)

    def compute_mask(max_length, curr_length):
        arange = torch.arange(max_length, device=device).repeat((embeddings.shape[0], 1)).permute(0, 1)
        lengths_stacked = curr_length.repeat((max_length, 1)).permute(1, 0)
        return arange <= lengths_stacked

    # compute masks
    mask_chord = compute_mask(max_num_chords, num_chords)
    mask_melody = compute_mask(max_num_notes, num_notes)

    loss_chords = torch.masked_select(loss_chords, mask_chord).mean()
    loss_melody = torch.masked_select(loss_melody_notes, mask_melody).mean()
    loss_melody *= 10
    loss_kl = kl * 1e-2
    loss_bpm = mae(pred_bpm[:, 0], bpm_gt) / 5
    loss_valence = mae(pred_valence[:, 0], valence_gt) / 5
    loss_energy = mae(pred_energy[:, 0], energy_gt) / 5
    loss_total = loss_chords + loss_kl + loss_melody + loss_bpm + loss_energy + loss_valence
    tp_chords = torch.masked_select(pred_chords.argmax(dim=2) == chords_gt, mask_chord).tolist()
    tp_melodies = torch.masked_select(pred_notes.argmax(dim=2) == notes_gt, mask_melody).tolist()

    return loss_total, loss_chords, loss_kl, loss_melody, loss_bpm, loss_valence, loss_energy, tp_chords, tp_melodies


if __name__ == '__main__':
    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)

    dataset_folder = "dataset/processed-lyrics-spotify"
    dataset_files = os.listdir(dataset_folder)
    embeddings_file = "embeddings"  # without .npy extension
    embedding_lengths_file = "embedding_lengths.json"

    dataset = SongDataset(dataset_folder, dataset_files, embeddings_file, embedding_lengths_file)
    train_size = int(0.8 * len(dataset))
    validation_size = len(dataset) - train_size
    train_dataset, validation_dataset = torch.utils.data.random_split(dataset, [train_size, validation_size])
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    validation_dataloader = DataLoader(validation_dataset, batch_size=BATCH_SIZE, shuffle=False)

    def count_map_to_weights(count_map):
        weights = torch.tensor(list(count_map.values()), dtype=torch.float32, device=device)
        weights = weights / weights.sum()
        weights = 1.0 / weights
        return weights / weights.sum()

    chord_weights = count_map_to_weights(dataset.chord_count_map)
    note_weights = count_map_to_weights(dataset.melody_note_count_map)

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
        epochs.append(epoch)

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

        # TRAINING
        model.train()
        for batch, data in enumerate(train_dataloader):
            loss, loss_chords, kl_loss, loss_melody, loss_bpm, loss_valence, loss_energy, batch_tp_chords, batch_tp_melodies = compute_loss(data)

            epoch_training_losses_chords.append(loss_chords)
            epoch_training_losses_melodies.append(loss_melody)
            epoch_training_losses_kl.append(kl_loss)
            epoch_training_tp_chords.extend(batch_tp_chords)
            epoch_training_tp_melodies.extend(batch_tp_melodies)

            optimizer.zero_grad()
            loss.backward()

            optimizer.step()
            loss = loss.item()
            print(f"\tBatch {batch}: Loss {loss} (C: {loss_chords} + KL: {kl_loss} + M: {loss_melody} + B: {loss_bpm} V: {loss_valence} + E: {loss_energy}), batch chord acc. cum. {sum(epoch_training_tp_chords) / len(epoch_training_tp_chords)}, batch melody acc. cum. {sum(epoch_training_tp_melodies) / len(epoch_training_tp_melodies)}")

        # VALIDATION
        model.eval()
        for batch, data in enumerate(validation_dataloader):
            with torch.no_grad():
                loss, loss_chords, kl_loss, loss_melody, loss_bpm, loss_valence, loss_energy, batch_tp_chords, batch_tp_melodies = compute_loss(data)

                epoch_validation_losses_chords.append(loss_chords)
                epoch_validation_losses_melodies.append(loss_melody)
                epoch_validation_losses_kl.append(kl_loss)
                epoch_validation_tp_chords.extend(batch_tp_chords)
                epoch_validation_tp_melodies.extend(batch_tp_melodies)

                print(f"\tVALIDATION Batch {batch}: Loss {loss} (C: {loss_chords} + KL: {kl_loss} + M: {loss_melody} + B: {loss_bpm} V: {loss_valence} + E: {loss_energy}), batch chord acc. cum. {sum(epoch_validation_tp_chords) / len(epoch_validation_tp_chords)}, batch melody acc. cum. {sum(epoch_validation_tp_melodies) / len(epoch_validation_tp_melodies)}")

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
        axs[0, 0].legend()
        axs[0, 0].grid(True)
        # Chords accuracy
        axs[1, 0].set_title('Chords accuracy')
        axs[1, 0].plot(epochs, training_accuracies_chords, label='Train', color='darkorange')
        axs[1, 0].plot(epochs, validation_accuracies_chords, label='Val', color='darkorange', linestyle='dotted')
        axs[1, 0].set_xlabel('Epochs')
        axs[1, 0].set_ylabel('Accuracy (%)')
        axs[1, 0].set_ylim(bottom=0)
        axs[1, 0].legend()
        axs[1, 0].grid(True)
        # Melody loss
        axs[0, 1].set_title('Melody loss')
        axs[0, 1].plot(epochs, training_losses_melodies, label='Train', color='royalblue')
        axs[0, 1].plot(epochs, validation_losses_melodies, label='Val', color='royalblue', linestyle='dotted')
        axs[0, 1].set_xlabel('Epochs')
        axs[0, 1].set_ylabel('Loss')
        axs[0, 1].legend()
        axs[0, 1].grid(True)
        # Melody accuracy
        axs[1, 1].set_title('Melody accuracy')
        axs[1, 1].plot(epochs, training_accuracies_melodies, label='Train', color='darkorange')
        axs[1, 1].plot(epochs, validation_accuracies_melodies, label='Val', color='darkorange', linestyle='dotted')
        axs[1, 1].set_xlabel('Epochs')
        axs[1, 1].set_ylabel('Accuracy (%)')
        axs[1, 1].set_ylim(bottom=0)
        axs[1, 1].legend()
        axs[1, 1].grid(True)
        plot.tight_layout()
        plot.show()
