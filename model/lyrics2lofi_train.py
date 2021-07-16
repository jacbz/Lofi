import os

from model.lyrics2lofi_dataset import Lyrics2LofiDataset
from model.lyrics2lofi_model import Lyrics2LofiModel
from model.train import train

if __name__ == '__main__':
    dataset_folder = "dataset/processed-lyrics-spotify"
    dataset_files = os.listdir(dataset_folder)
    embeddings_file = "embeddings"  # without .npy extension
    embedding_lengths_file = "embedding_lengths.json"

    dataset = Lyrics2LofiDataset(dataset_folder, dataset_files, embeddings_file, embedding_lengths_file)
    model = Lyrics2LofiModel()

    train(dataset, model, "lyrics2lofi")
