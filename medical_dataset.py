import os
import nibabel as nib
import numpy as np
import torch.utils.data
import json
import glob


class MedicalDataset(torch.utils.data.Dataset):
    def __init__(self, config):
        self.config = json.load(open(config))

        self.data = None
        self._read_data()

    def _read_data(self):
        data = []
        for path in glob.glob(os.path.join(self.config['pretrain_data_path'], "*.nii")):
            data.append(np.transpose(nib.load(path).get_fdata(), (2, 0, 1)))

        self.data = data

    def __getitem__(self, index):
        data = np.expand_dims(self.data[index][0], 0)
        return data, data

    def __len__(self):
        return len(self.data)

    def __repr__(self):
        return self.__class__.__name__
