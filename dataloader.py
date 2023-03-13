import random
import librosa
from torch.utils.data import Dataset, DataLoader
import time
import utils
import numpy as np

import scipy.io.wavfile as wavfile
import numpy as np


class MyDataset(Dataset):
    def __init__(self, file_path):
        with open(file_path, "r") as f:
            self.filenames = [line.strip() for line in f.readlines()]
            # self.epochs = epochs
            # self.batch_size = batch_size

    def __len__(self):
        return len(self.filenames) // 2
        # return len(self.filenames) // (2 * self.batch_size)

    def __getitem__(self, index):
        # epoch_index = index // self.__len__()
        # seed_value= (epoch_index * self.epochs + index % self.__len__())
        # random_ints = utils.generate_random_ints(str(seed_value), self.__len__())   # use epoch batch size and index to generate seed value.
        # video_file = "videoFeatures/" + self.filenames[random_ints[0]] + ".npy"
        # video = np.load(video_file)
        # filename1 = "audioFeatures/" + self.filenames[random_ints[0]] + ".npy"
        # filename2 = "audioFeatures/" + self.filenames[random_ints[1] + 1] + ".npy"

        video_file = "videoFeatures/" + self.filenames[index] + ".npy"
        video = np.load(video_file)
        filename1 = "audioFeatures/" + self.filenames[index] + ".npy"
        filename2 = "audioFeatures/" + self.filenames[index + 1] + ".npy"

        F_mix, cRM1 = self.__audioGeneration(filename1, filename2)

        return [video.astype(np.float32), F_mix.astype(np.float32)], cRM1.astype(
            np.float32
        )

    def __audioGeneration(self, path1: str, path2: str):
        """
        :param path1: first file path
        :param path2: second  file path
        :return: F_mix,cRM1,cRM2
        """
        data1 = np.load(path1)
        data2 = np.load(path2)

        # Apply fast_stft on the mix track
        F_mix = utils.fast_stft(data1 + data2)

        # Apply fast_stft on individual tracks
        F_single1 = utils.fast_stft(data1)

        # Generate the CRM of each track
        cRM1 = utils.fast_cRM(F_single1, F_mix)
        return F_mix, cRM1
