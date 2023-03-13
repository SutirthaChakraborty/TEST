import os
import utils
import torch
import shutil
import warnings
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from model import RealTimeAVModel
import scipy.io.wavfile as wavfile
from dataloader import DataLoader, MyDataset

warnings.filterwarnings("ignore")
import os

# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"]="1"
import numpy as np
import mir_eval


def smooth_signal(signal):
    smoothed_signal = signal.copy()

    for i, value in enumerate(smoothed_signal):
        if np.isnan(value) or np.isinf(value):
            # Find the indices of the previous and next non-NaN/non-inf values
            prev_index = i - 1
            while prev_index >= 0 and (
                np.isnan(smoothed_signal[prev_index])
                or np.isinf(smoothed_signal[prev_index])
            ):
                prev_index -= 1
            next_index = i + 1
            while next_index < len(smoothed_signal) and (
                np.isnan(smoothed_signal[next_index])
                or np.isinf(smoothed_signal[next_index])
            ):
                next_index += 1

            # If both previous and next values are available, set the current value to their mean
            if prev_index >= 0 and next_index < len(smoothed_signal):
                smoothed_signal[i] = (
                    smoothed_signal[prev_index] + smoothed_signal[next_index]
                ) / 2.0

    return smoothed_signal


sdr_List = []
sir_List = []
sar_List = []

# specify the path to the folder to be deleted
folder_path = "output1"
if os.path.exists(folder_path):
    # delete the folder and its contents using shutil.rmtree()
    shutil.rmtree(folder_path)
# create a new empty folder using os.makedirs()
os.makedirs(folder_path)


# Set the model to evaluation mode
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
# Load the saved model and optimizer state
load_path = "best_model_state_dict.pth"
# load_path = "savedModels/batch_8_epochs_100_lr_0.001.pth"
checkpoint = torch.load(load_path)
model = RealTimeAVModel().to(device)
# Define the optimizer
optimizer = optim.Adam(model.parameters(), lr=1e-3)
model.load_state_dict(checkpoint["model_state_dict"])
optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
model.eval()
batch_size = 1

# Define the data loader for inference
inference_dataset = MyDataset("test.txt")
inference_dataloader = DataLoader(
    inference_dataset, batch_size=batch_size, shuffle=False
)
i = 0
# Iterate over the inference data and perform inference

for inputs, targets in inference_dataloader:
    inputs_numpy = [inp.to(device) for inp in inputs]
    video, F_mix = inputs_numpy[0], inputs_numpy[1]

    # Mix Audio
    T_mix = utils.fast_istft(F_mix[0].cpu().detach().numpy(), power=False)
    filename = folder_path + "/" + str(i) + "_mix.wav"
    wavfile.write(filename, 24000, T_mix)

    # Ground truth audio
    tar = targets[0].cpu().detach().numpy()  # checking
    F = utils.fast_icRM(F_mix[0].cpu().detach().numpy(), tar)
    T_original = utils.fast_istft(F, power=False)
    filename = folder_path + "/" + str(i) + "_clean.wav"
    wavfile.write(filename, 24000, T_original)

    # Predicted Sound
    outputs = model(inputs_numpy)
    F_mix = F_mix[0].cpu().detach().numpy()
    y_hat = outputs[0].cpu().detach().numpy()
    F = utils.fast_icRM(F_mix, y_hat)
    T_prediction = utils.fast_istft(F, power=False)
    filename = folder_path + "/" + str(i) + ".wav"
    wavfile.write(filename, 24000, T_prediction)
    i += 1
    (sdr, sir, sar, perm) = mir_eval.separation.bss_eval_sources(
        smooth_signal(T_original), smooth_signal(T_prediction)
    )

    sdr_List.append(sdr)
    sar_List.append(sar)


filename = "metric.txt"


with open(filename, "a") as file:
    file.write(
        f"File= {load_path},    SDR = {sum(sdr_List)/len(sdr_List)},     SAR = {sum(sar_List)/len(sar_List)}\n"
    )
