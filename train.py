import os
import torch
import datetime
import warnings
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from model import RealTimeAVModel
from loss_paper import mse_loss_for_variable_length_data
from dataloader import DataLoader, MyDataset
from torch.utils.tensorboard import SummaryWriter

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
warnings.filterwarnings("ignore")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
# device = torch.device("cpu")
# Initialize the model
model = RealTimeAVModel().to(device)

# Define the loss function
mse_loss = nn.MSELoss()
loss_function = mse_loss_for_variable_length_data()


# Define the optimizer
lr = 1e-3
batch_size = 1
# Train the model for N epochs
num_epochs = 100


optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Define the data loaders for training and validation

train_dataset = MyDataset("train.txt")
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataset = MyDataset("test.txt")
val_dataloader = DataLoader(val_dataset, batch_size=batch_size)


# Define the TensorBoard writer and the global step
global_step = 0

# Define the best validation loss and the corresponding epoch number
best_val_loss = float("inf")
best_val_epoch = -1


# Set the model to training mode
model.train()

now = datetime.datetime.now()
timestamp_str = now.strftime(
    "%Y-%m-%d_%H-%M-%S"
)  # generate timestamp string in the format of "YYYY-MM-DD_HH-MM-SS"

log_dir = f"/warm-data/avss/logs/separateloss_{timestamp_str}_batch_size_{batch_size}_epochs_{num_epochs}_lr_{lr}"
writer = SummaryWriter(log_dir=log_dir)

num_frames = 120
n_frames_list = [num_frames] * batch_size

for epoch in range(num_epochs):
    # Train for one epoch
    running_train_loss = 0.0
    for i, (inputs, targets) in enumerate(train_dataloader):
        inputs = [inp.to(device) for inp in inputs]
        targets = targets.to(device)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)

        # Calculate the loss
        # loss = mse_loss(outputs, targets)
        loss = loss_function(
            targets[:, :, :, 0], outputs[:, :, :, 0], n_frames_list, device
        ) + loss_function(
            targets[:, :, :, 1], outputs[:, :, :, 1], n_frames_list, device
        )

        # Backward pass
        loss.backward()

        # Update the parameters
        optimizer.step()

        running_train_loss += loss.item()

    # Calculate validation loss
    running_val_loss = 0.0
    with torch.no_grad():
        for i, (inputs, targets) in enumerate(val_dataloader):
            inputs = [inp.to(device) for inp in inputs]
            targets = targets.to(device)
            outputs = model(inputs)
            # loss = mse_loss(outputs, targets)
            loss = loss_function(
                targets[:, :, :, 0], outputs[:, :, :, 0], n_frames_list, device
            ) + loss_function(
                targets[:, :, :, 1], outputs[:, :, :, 1], n_frames_list, device
            )

            running_val_loss += loss.item()
            # Log the batch loss to TensorBoard
            writer.add_scalar("Training Loss", loss.item(), global_step)
            global_step += 1

    val_loss = running_val_loss / len(val_dataloader)
    # Log the epoch losses to TensorBoard

    # Calculate epoch loss and print progress
    epoch_train_loss = running_train_loss / len(train_dataloader)
    print(
        f"Epoch {epoch+1}/{num_epochs}, Train Loss: {epoch_train_loss:.4f}, Val Loss: {val_loss:.4f}"
    )
    writer.add_scalar("Epoch Training Loss", epoch_train_loss, epoch)
    writer.add_scalar("Epoch Validation Loss", val_loss, epoch)

    # Save the model with the best validation loss
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        save_path = "best_model_state_dict.pth"
        # save_path ="savedModels/batch_" +str(batch_size)+"_epochs_"+str(num_epochs)+"_lr_"+str(lr)+".pth"
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            },
            save_path,
        )
