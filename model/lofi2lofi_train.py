import os

from model.lofi2lofi_dataset import Lofi2LofiDataset
from model.lofi2lofi_model import Lofi2LofiModel
from model.train import train

if __name__ == '__main__':
    dataset_folder = "dataset/processed-spotify-all"
    dataset_files = os.listdir(dataset_folder)

    dataset = Lofi2LofiDataset(dataset_folder, dataset_files)
    model = Lofi2LofiModel()

    train(dataset, model, "lofi2lofi")
