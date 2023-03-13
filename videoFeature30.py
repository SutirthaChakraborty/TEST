import os
import glob
import warnings
import numpy as np

warnings.filterwarnings("ignore")
import cv2
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from retinaface import RetinaFace
import librosa
import shutil
from moviepy.editor import VideoFileClip
import utils

# specify the path to the folder to be deleted
folder_path = "videoFeatures"
if os.path.exists(folder_path):
    # delete the folder and its contents using shutil.rmtree()
    shutil.rmtree(folder_path)
# create a new empty folder using os.makedirs()
os.makedirs(folder_path)

# specify the path to the folder to be deleted
folder_path = "audioFeatures"
if os.path.exists(folder_path):
    # delete the folder and its contents using shutil.rmtree()
    shutil.rmtree(folder_path)
# create a new empty folder using os.makedirs()
os.makedirs(folder_path)


# Define a MobileNetV2-based feature extractor
class MobileNetV2FeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        mobilenet = models.mobilenet_v2(pretrained=True)
        self.features = nn.Sequential(*list(mobilenet.children())[:-1])
        self.conv = nn.Conv2d(1280, 16, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x = self.features(x)
        x = self.conv(x)
        x = F.adaptive_avg_pool2d(x, 1).squeeze()
        return x


# Create an instance of the MobileNetV2-based feature extractor
feature_extractor = MobileNetV2FeatureExtractor()
feature_extractor.eval()


def videoFeature(filename):
    try:
        clip = VideoFileClip(filename)
        # Define the new FPS
        new_fps = 30
        # Resample the clip to the new FPS
        resampled_clip = clip.set_fps(new_fps)
        resampled_clip.write_videofile("temp.mp4", verbose=False, logger=None)

        # Open the video file
        cap = cv2.VideoCapture("temp.mp4")

        # Create an empty list to store features
        features_list = []

        # Extract features for the first 30 frames
        for i in range(30*utils.SECONDS):
            # Read a frame from the video
            ret, frame = cap.read()
            if not ret:
                break

            # Extract faces using RetinaFace
            faces = RetinaFace.extract_faces(frame, align=True)[0]

            # Resize the face image to 224x224 and convert it to a tensor
            resized_face = cv2.resize(faces, (224, 224))
            img_tensor = (
                torch.tensor(resized_face).permute(2, 0, 1).unsqueeze(0).float() / 255.0
            )

            # Extract features using MobileNetV2-based feature extractor
            with torch.no_grad():
                features = feature_extractor(img_tensor)
                features = features.numpy()

            # Append the features to the list
            features_list.append(features)
            features_list.append(features)
            features_list.append(features)
            features_list.append(features)

        # Release the video capture object and close all windows
        cap.release()

        # #print the length of the features list
        return np.array(features_list), True
    except:
        return [], False

files = glob.glob("/warm-data/nassau-avss/video_train/*.mp4")[0:50]  # TODO : change the limit to experiment with number of files 

def audioFeatures(filename):
    try:
        fix_sr = 24000
        data, _ = librosa.load(filename, sr=fix_sr, duration=utils.SECONDS)
        data = data[:24000*utils.SECONDS]
        max = np.max(np.abs(data))
        data = np.divide(data, max)
        return data, True
    except:
        return [], False


import tqdm

for i in tqdm.tqdm(
    files,
    total=len(files),
    desc="Processing files",
    bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt}",
    mininterval=0.1,
):
    videofeatures, status_video = videoFeature(i)
    audiofeatures, status_audio = audioFeatures(i)
    # print(i)
    if status_video and status_audio:
        filename = i.replace("/warm-data/nassau-avss/video_train/", "videoFeatures/")
        filename = filename.replace("mp4", "npy")
        np.save(filename, videofeatures)

        filename = i.replace("/warm-data/nassau-avss/video_train/", "audioFeatures/")
        filename = filename.replace("mp4", "npy")
        np.save(filename, audiofeatures)
