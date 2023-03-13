import random
import glob

folder_path = "videoFeatures/"
file_extension = ".npy"
files = glob.glob(folder_path + "*")
file = open("files.txt", "w")
for i in files:
    p = i[i.rindex("/") + 1 : i.rindex(".")]
    file.write(p + "\n")
file.close()
input_file = "files.txt"
train_file = "train.txt"
test_file = "test.txt"

# Set the desired train-test ratio
train_ratio = 0.8

# Read the lines from the input file
with open(input_file, "r") as f:
    lines = f.readlines()

lines = lines + lines + lines + lines

# Shuffle the lines randomly
random.shuffle(lines)

# Determine the index at which to split the lines into train and test sets
split_index = int(len(lines) * train_ratio)

# Write the lines to the output files
with open(train_file, "w") as f:
    f.writelines(lines[:split_index])

with open(test_file, "w") as f:
    f.writelines(lines[split_index:])
