import matplotlib.pyplot as plot
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.data import DataLoader

from model.constants import *


def train(dataset, model, name):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")

    train_size = int(TRAIN_VALIDATION_SPLIT * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    ce_loss = nn.CrossEntropyLoss(reduction='none')
    l1_loss = nn.L1Loss(reduction='mean')

    model = model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    epochs = []
    train_losses_chords, train_losses_melodies, train_losses_kl, train_accs_chords, train_accs_melodies = [], [], [], [], []
    val_losses_chords, val_losses_melodies, val_losses_kl, val_accs_chords, val_accs_melodies = [], [], [], [], []

    # losses for one batch of data
    def compute_loss(data):
        if name == "lyrics2lofi":
            embeddings = data["embedding"].to(device)
            embedding_lengths = data["embedding_length"]

        num_chords = data["num_chords"]
        max_num_chords = num_chords.max()
        max_num_notes = max_num_chords * NOTES_PER_CHORD

        chords_gt = data["chords"].to(device)[:, :max_num_chords]
        notes_gt = data["melody_notes"].to(device)[:, :max_num_notes]
        tempo_gt = data["tempo"].to(device)
        key_gt = data["key"].to(device)
        mode_gt = data["mode"].to(device)
        valence_gt = data["valence"].to(device)
        energy_gt = data["energy"].to(device)

        # run model
        if name == "lyrics2lofi":
            input = pack_padded_sequence(embeddings, embedding_lengths, batch_first=True, enforce_sorted=False)
            pred_chords, pred_notes, pred_tempo, pred_key, pred_mode, pred_valence, pred_energy, kl = \
                model(input, max_num_chords, sampling_rate_chords, sampling_rate_melodies, chords_gt, notes_gt)
        else:
            pred_chords, pred_notes, pred_tempo, pred_key, pred_mode, pred_valence, pred_energy, kl = \
                model(chords_gt, notes_gt, tempo_gt, key_gt, mode_gt, valence_gt, energy_gt, num_chords, max_num_chords,
                      sampling_rate_chords, sampling_rate_melodies)

        # compute a boolean mask to select entries up to a specific index
        def compute_mask(max_length, curr_length):
            arange = torch.arange(max_length, device=device).repeat((chords_gt.shape[0], 1)).permute(0, 1)
            lengths_stacked = curr_length.repeat((max_length, 1)).permute(1, 0)
            return arange <= lengths_stacked

        num_chords = num_chords.to(device)
        loss_chords = ce_loss(pred_chords.permute(0, 2, 1), chords_gt)
        mask_chord = compute_mask(max_num_chords, num_chords)
        loss_chords = torch.masked_select(loss_chords, mask_chord).mean()

        num_notes = num_chords * NOTES_PER_CHORD
        loss_melody_notes = ce_loss(pred_notes.permute(0, 2, 1), notes_gt)
        mask_melody = compute_mask(max_num_notes, num_notes)
        loss_melody = torch.masked_select(loss_melody_notes, mask_melody).mean()

        if epoch < MELODY_EPOCH_DELAY:
            loss_melody = 0

        loss_kl = kl
        loss_tempo = l1_loss(pred_tempo[:, 0], tempo_gt) / 5
        loss_key = ce_loss(pred_key, key_gt).mean() / 30
        loss_mode = ce_loss(pred_mode, mode_gt).mean() / 10
        loss_valence = l1_loss(pred_valence[:, 0], valence_gt) / 5
        loss_energy = l1_loss(pred_energy[:, 0], energy_gt) / 5
        loss_total = loss_chords + loss_kl + loss_melody + loss_tempo + loss_key + loss_mode + loss_energy + loss_valence

        tp_chords = torch.masked_select(pred_chords.argmax(dim=2) == chords_gt, mask_chord).tolist()
        tp_melodies = torch.masked_select(pred_notes.argmax(dim=2) == notes_gt, mask_melody).tolist()

        return loss_total, loss_chords, loss_kl, loss_melody, loss_tempo, loss_key, loss_mode, loss_valence, loss_energy, tp_chords, tp_melodies

    print(f"Starting training: {name}")
    epoch = 0
    while True:
        epochs.append(epoch)

        print(f"== Epoch {epoch} ==")
        ep_train_losses_chords, ep_train_losses_melodies, ep_train_losses_kl, ep_train_tp_chords, ep_train_tp_melodies = [], [], [], [], []
        ep_val_losses_chords, ep_val_losses_melodies, ep_val_losses_kl, ep_val_tp_chords, ep_val_tp_melodies = [], [], [], [], []

        sampling_rate_chords = 0
        sampling_rate_melodies = 0

        if TEACHER_FORCE:
            sampling_rate_chords = sampling_rate_at_epoch(epoch)
            sampling_rate_melodies = sampling_rate_at_epoch(epoch - MELODY_EPOCH_DELAY)

        print(f"Scheduled sampling rate: C {sampling_rate_chords}, M {sampling_rate_melodies}")

        # TRAINING
        model.train()
        for batch, data in enumerate(train_dataloader):
            loss, loss_chords, kl_loss, loss_melody, \
            loss_tempo, loss_key, loss_mode, loss_valence, loss_energy, \
            batch_tp_chords, batch_tp_melodies = compute_loss(data)

            ep_train_losses_chords.append(loss_chords)
            ep_train_losses_melodies.append(loss_melody)
            ep_train_losses_kl.append(kl_loss)
            ep_train_tp_chords.extend(batch_tp_chords)
            ep_train_tp_melodies.extend(batch_tp_melodies)

            optimizer.zero_grad()
            loss.backward()

            optimizer.step()
            loss = loss.item()
            print(f"\tBatch {batch}:\tLoss {loss:.3f} (C: {loss_chords:.3f} + KL: {kl_loss:.3f} + "
                  f"M: {loss_melody:.3f} + T: {loss_tempo:.3f} + K: {loss_key:.3f} + Mo: {loss_mode:.3f} + "
                  f"V: {loss_valence:.3f} + E: {loss_energy:.3f})")

        # VALIDATION
        model.eval()
        for batch, data in enumerate(val_dataloader):
            with torch.no_grad():
                loss, loss_chords, kl_loss, loss_melody, \
                loss_tempo, loss_key, loss_mode, loss_valence, loss_energy, \
                batch_tp_chords, batch_tp_melodies = compute_loss(data)

                ep_val_losses_chords.append(loss_chords)
                ep_val_losses_melodies.append(loss_melody)
                ep_val_losses_kl.append(kl_loss)
                ep_val_tp_chords.extend(batch_tp_chords)
                ep_val_tp_melodies.extend(batch_tp_melodies)

                print(f"\tValidation Batch {batch}:\tLoss {loss:.3f} (C: {loss_chords:.3f} + KL: {kl_loss:.3f} + "
                      f"M: {loss_melody:.3f} + T: {loss_tempo:.3f} + K: {loss_key:.3f} + Mo: {loss_mode:.3f} + "
                      f"V: {loss_valence:.3f} + E: {loss_energy:.3f})")

        # copy old model
        save_name = f"{name}-epoch{epoch}.pth" if epoch % 10 == 0 else f"{name}.pth"
        decoder_save_name = f"{name}-decoder-epoch{epoch}.pth" if epoch % 10 == 0 else f"{name}-decoder.pth"
        torch.save(model.state_dict(), save_name)
        torch.save(model.decoder.state_dict(), decoder_save_name)
        epoch += 1

        ep_train_loss_chord = sum(ep_train_losses_chords) / len(ep_train_losses_chords)
        ep_train_loss_melody = sum(ep_train_losses_melodies) / len(ep_train_losses_melodies)
        ep_train_loss_kl = sum(ep_train_losses_kl) / len(ep_train_losses_kl)
        ep_train_chord_acc = (sum(ep_train_tp_chords) / len(ep_train_tp_chords)) * 100
        ep_train_melody_acc = (sum(ep_train_tp_melodies) / len(ep_train_tp_melodies)) * 100

        ep_val_loss_chord = sum(ep_val_losses_chords) / len(ep_val_losses_chords)
        ep_val_loss_melody = sum(ep_val_losses_melodies) / len(ep_val_losses_melodies)
        ep_val_loss_kl = sum(ep_val_losses_kl) / len(ep_val_losses_kl)
        ep_val_chord_acc = (sum(ep_val_tp_chords) / len(ep_val_tp_chords)) * 100
        ep_val_melody_acc = (sum(ep_val_tp_melodies) / len(ep_val_tp_melodies)) * 100

        print(
            f"Epoch chord loss: {ep_train_loss_chord:.3f}, melody loss: {ep_train_loss_melody:.3f}, KL: {ep_train_loss_kl:.3f}, "
            f"chord accuracy: {ep_train_chord_acc:.3f}, melody accuracy: {ep_train_melody_acc:.3f}")
        print(
            f"VALIDATION: epoch chord loss: {ep_val_loss_chord:.3f}, melody loss: {ep_val_loss_melody:.3f}, KL: {ep_val_loss_kl:.3f}, "
            f"chord accuracy: {ep_val_chord_acc:.3f}, melody accuracy: {ep_val_melody_acc:.3f}")

        train_losses_chords.append(ep_train_loss_chord)
        train_losses_melodies.append(ep_train_loss_melody)
        train_losses_kl.append(ep_train_loss_kl)
        train_accs_chords.append(ep_train_chord_acc)
        train_accs_melodies.append(ep_train_melody_acc)

        val_losses_chords.append(ep_val_loss_chord)
        val_losses_melodies.append(ep_val_loss_melody)
        val_losses_kl.append(ep_val_loss_kl)
        val_accs_chords.append(ep_val_chord_acc)
        val_accs_melodies.append(ep_val_melody_acc)

        fig, axs = plot.subplots(2, 2, figsize=(8, 4.5), dpi=200)
        # Chords loss
        axs[0, 0].set_title('Chords loss')
        axs[0, 0].plot(epochs, train_losses_chords, label='Train', color='royalblue')
        axs[0, 0].plot(epochs, val_losses_chords, label='Val', color='royalblue', linestyle='dotted')
        axs[0, 0].set_xlabel('Epochs')
        axs[0, 0].set_ylabel('Loss')
        axs[0, 0].legend()
        axs[0, 0].grid(True)
        # Chords accuracy
        axs[1, 0].set_title('Chords accuracy')
        axs[1, 0].plot(epochs, train_accs_chords, label='Train', color='darkorange')
        axs[1, 0].plot(epochs, val_accs_chords, label='Val', color='darkorange', linestyle='dotted')
        axs[1, 0].set_xlabel('Epochs')
        axs[1, 0].set_ylabel('Accuracy (%)')
        axs[1, 0].set_ylim(bottom=0)
        axs[1, 0].legend()
        axs[1, 0].grid(True)
        # Melody loss
        axs[0, 1].set_title('Melody loss')
        axs[0, 1].plot(epochs, train_losses_melodies, label='Train', color='royalblue')
        axs[0, 1].plot(epochs, val_losses_melodies, label='Val', color='royalblue', linestyle='dotted')
        axs[0, 1].set_xlabel('Epochs')
        axs[0, 1].set_ylabel('Loss')
        axs[0, 1].legend()
        axs[0, 1].grid(True)
        # Melody accuracy
        axs[1, 1].set_title('Melody accuracy')
        axs[1, 1].plot(epochs, train_accs_melodies, label='Train', color='darkorange')
        axs[1, 1].plot(epochs, val_accs_melodies, label='Val', color='darkorange', linestyle='dotted')
        axs[1, 1].set_xlabel('Epochs')
        axs[1, 1].set_ylabel('Accuracy (%)')
        axs[1, 1].set_ylim(bottom=0)
        axs[1, 1].legend()
        axs[1, 1].grid(True)

        plot.tight_layout()
        plot.savefig(f"{name}.png")
        plot.show()
