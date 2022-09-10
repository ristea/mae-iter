import os
import nibabel as nib
import numpy as np
import pandas as pd
import torch.utils.data
import json
import glob
import pydicom
import cv2


class MedicalDataset(torch.utils.data.Dataset):
    def __init__(self, config, transform=None):
        self.config = json.load(open(config))
        self.transform = transform
        self.data = []

        training_data_sets = self.config["dataset-pretrain"]
        if "coltea-db-path" in training_data_sets:
            self._read_coltea()
        if "heart" in training_data_sets:
            raise Exception("Not implemented yet!")
        if "brats" in training_data_sets:
            self._read("BraTS", "MRI")
        if "hepaticvessel" in training_data_sets:
            self._read("HepaticVessel", "CT")
        if "hippocampus" in training_data_sets:
            self._read("Hippocampus", "MRI")
        if "lits" in training_data_sets:
            self._read("LITS", "CT")
        if "pancreas" in training_data_sets:
            self._read("Pancreas", "CT")
        if "prostate" in training_data_sets:
            self._read("Prostate", "MRI")
        if "spleen" in training_data_sets:
            self._read("Spleen", "CT")

    def _read(self, name, type):
        prods_path = []
        for path in glob.glob(os.path.join(self.config['pretrain_data_path'], name, "training", "images", "*.nii")):
            if name == "Prostate":
                length = nib.load(path).get_fdata().shape[-2]
            else:
                length = nib.load(path).get_fdata().shape[-1]

            for i in range(0, length - self.config['depth']):
                prods_path.append({
                    "data": path,
                    "slice": list(range(i, i + self.config['depth'])),
                    "type": type,
                    "organ": name
                })
        self.data += prods_path

    def _read_coltea(self):
        csv = pd.read_csv(os.path.join(self.config["coltea-db-path"], "train_data.csv"))
        data = []
        for dir_name in csv["train"]:
            slices = list(glob.glob(os.path.join(self.config["coltea-db-path"], "Coltea-Lung-CT-100W", dir_name, "*", "DICOM", "*")))
            data += slices
        data.sort()

        data = [{"data": x, "type": "CT"} for x in data]
        self.data += data

    def _load_data(self, meta):
        if meta['organ'] == "Prostate":
            image = nib.load(meta['data']).get_fdata()[:, :, meta['slice'], 0]
        else:
            image = nib.load(meta['data']).get_fdata()[:, :, meta['slice']]
        return image

    def __getitem__(self, index):
        meta = self.data[index]

        if meta['type'] == "CT":
            # image = pydicom.dcmread(meta['data']).pixel_array
            image = self._load_data(meta)
            image[image < 0] = 0
            image = image / 1e3
            image = image.astype(np.float32)
        elif meta['type'] == "MRI":
            image = self._load_data(meta)
            image[image < 0] = 0
            image = image / 1000
            image = image.astype(np.float32)

        image = cv2.resize(image, dsize=(256, 256), interpolation=cv2.INTER_CUBIC)
        image = np.moveaxis(image, -1, 0)

        if self.transform:
            image = self.transform(torch.from_numpy(image))

        return image, image

    def __len__(self):
        return len(self.data)

    def __repr__(self):
        return self.__class__.__name__
