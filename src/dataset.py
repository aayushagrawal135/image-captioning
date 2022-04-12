import os
import h5py
import json
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T


class FlickrDataset(Dataset):

    def __init__(self, data_folder: str, data_name: str, split: str, transform: T):

        self.split = split
        assert self.split in {"TRAIN", "VAL", "TEST"}

        image_filename = os.path.join(data_folder, f"{self.split}_IMAGES_{data_name}.hdf5")
        self.h = h5py.File(image_filename, "r")

        self.images = self.h["images"]
        self.captions_per_image = self.h.attrs["captions_per_image"]

        captions_filename = os.path.join(data_folder, f"{self.split}_CAPTIONS_{data_name}.json")
        with open(captions_filename, "r") as f:
            self.captions = json.load(f)

        caplens_filename = os.path.join(data_folder, f"{self.split}_CAPLENS_{data_name}.json")
        with open(caplens_filename, "r") as f:
            self.caplens = json.load(f)

        self.transform = transform

    def __getitem__(self, idx: int):

        image = torch.Tensor(self.images[idx // self.captions_per_image] / 255.)

        if self.transform is not None:
            image = self.transform(image)

        caption = torch.LongTensor(self.captions[idx])
        caplen = torch.LongTensor([self.caplens[idx]])

        if self.split == "TRAIN":
            return image, caption, caplen

        else:
            index = (idx // self.captions_per_image) * self.captions_per_image
            all_captions = self.captions[index: index + self.captions_per_image]
            all_captions = torch.LongTensor(all_captions)
            return image, caption, caplen, all_captions

    def __len__(self):
        return len(self.captions)
