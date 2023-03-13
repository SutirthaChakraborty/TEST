import torch.nn as nn
import torch.nn.functional as F
import torch
import os


class RealTimeAVModel(nn.Module):
    def __init__(self):
        super(RealTimeAVModel, self).__init__()

        # Encoder
        self.conv1a = nn.Conv2d(
            2, 16, kernel_size=(1, 3), padding=(0, 1), stride=(1, 2)
        )
        self.bn1a = nn.BatchNorm2d(16)
        self.relu1a = nn.ReLU()

        self.conv2a = nn.Conv2d(
            16, 16, kernel_size=(1, 3), padding=(0, 1), stride=(1, 2)
        )
        self.bn2a = nn.BatchNorm2d(16)
        self.relu2a = nn.ReLU()

        self.conv3a = nn.Conv2d(
            16, 16, kernel_size=(1, 3), padding=(0, 1), stride=(1, 2)
        )
        self.bn3a = nn.BatchNorm2d(16)
        self.relu3a = nn.ReLU()

        self.conv4a = nn.Conv2d(
            16, 16, kernel_size=(1, 3), padding=(0, 1), stride=(1, 2)
        )
        self.bn4a = nn.BatchNorm2d(16)
        self.relu4a = nn.ReLU()

        self.conv5a = nn.Conv2d(
            16, 16, kernel_size=(1, 3), padding=(0, 1), stride=(1, 2)
        )
        self.bn5a = nn.BatchNorm2d(16)
        self.relu5a = nn.ReLU()

        self.conv6a = nn.Conv2d(
            16, 16, kernel_size=(1, 3), padding=(0, 1), stride=(1, 2)
        )
        self.bn6a = nn.BatchNorm2d(16)
        self.relu6a = nn.ReLU()

        # GRU1
        self.gru1 = nn.GRU(64 + 16, 64, batch_first=True)

        # GRU2
        self.gru2 = nn.GRU(64, 64, batch_first=True)

        # Decoder
        self.deconv6a = nn.ConvTranspose2d(
            32, 16, kernel_size=(1, 3), padding=(0, 1), stride=(1, 2)
        )
        self.bn6b = nn.BatchNorm2d(32)
        self.relu6b = nn.ReLU()

        self.deconv5a = nn.ConvTranspose2d(
            32, 16, kernel_size=(1, 3), padding=(0, 1), stride=(1, 2)
        )
        self.bn5b = nn.BatchNorm2d(32)
        self.relu5b = nn.ReLU()

        self.deconv4a = nn.ConvTranspose2d(
            32,
            16,
            kernel_size=(1, 3),
            padding=(0, 1),
            stride=(1, 2),
            output_padding=(0, 1),
        )
        self.bn4b = nn.BatchNorm2d(32)
        self.relu4b = nn.ReLU()

        self.deconv3a = nn.ConvTranspose2d(
            32, 16, kernel_size=(1, 3), padding=(0, 1), stride=(1, 2)
        )
        self.bn3b = nn.BatchNorm2d(32)
        self.relu3b = nn.ReLU()

        self.deconv2a = nn.ConvTranspose2d(
            32, 16, kernel_size=(1, 3), padding=(0, 1), stride=(1, 2)
        )
        self.bn2b = nn.BatchNorm2d(32)
        self.relu2b = nn.ReLU()

        self.deconv1a = nn.ConvTranspose2d(
            32, 2, kernel_size=(1, 3), padding=(0, 1), stride=(1, 2)
        )
        self.bn1b = nn.BatchNorm2d(32)
        self.relu1b = nn.ReLU()

    def forward(self, inputs):
        video_feat, mix = inputs

        # print("hh= ",video_feat.shape)
        mix = mix.view(mix.shape[0], mix.shape[3], mix.shape[1], mix.shape[2])

        # Encoder
        conv1 = self.conv1a(mix)
        # print(conv1.shape)
        conv2 = self.conv2a(F.relu(self.bn1a(conv1)))
        # print(conv2.shape)
        conv3 = self.conv3a(F.relu(self.bn2a(conv2)))
        # print(conv3.shape)
        conv4 = self.conv4a(F.relu(self.bn3a(conv3)))
        # print(conv4.shape)
        conv5 = self.conv5a(F.relu(self.bn4a(conv4)))
        # print(conv5.shape)
        conv6 = self.conv6a(F.relu(self.bn5a(conv5)))
        # print(conv6.shape)

        mix_emb = conv6.permute(0, 2, 1, 3).reshape(conv6.shape[0], conv6.shape[2], -1)

        # GRU input
        # Concatenate visual features
        gru1_in = torch.cat((mix_emb, video_feat), dim=2)
        gru1_out, _ = self.gru1(gru1_in)

        # GRU2

        gru2_out, _ = self.gru2(gru1_out)

        gru2_out = gru2_out.reshape(gru2_out.size()[0], gru2_out.size()[1], 16, -1)
        gru2_out = gru2_out.permute(0, 2, 1, 3)

        # Decoder

        out1 = torch.cat((gru2_out, conv6), dim=1)

        deconv6 = self.deconv6a(F.relu(self.bn6b(out1)))
        out2 = torch.cat((deconv6, conv5), 1)
        # print(deconv6.shape)
        deconv5 = self.deconv5a(F.relu(self.bn5b(out2)))
        out3 = torch.cat((deconv5, conv4), 1)
        # print(deconv5.shape)
        deconv4 = self.deconv4a(F.relu(self.bn4b(out3)))
        out4 = torch.cat((deconv4, conv3), 1)
        # print(deconv4.shape)
        deconv3 = self.deconv3a(F.relu(self.bn3b(out4)))
        out5 = torch.cat((deconv3, conv2), 1)
        # print(deconv3.shape)
        deconv2 = self.deconv2a(F.relu(self.bn2b(out5)))
        out6 = torch.cat((deconv2, conv1), 1)
        # print(deconv2.shape)
        deconv1 = self.deconv1a(F.relu(self.bn1b(out6)))
        # print(deconv1.shape)
        deconv1 = deconv1.squeeze(dim=1)

        # Output
        s = nn.Sigmoid()
        output = s(deconv1)

        # print("--",output.shape)
        output = output.view(
            output.shape[0], output.shape[2], output.shape[3], output.shape[1]
        )

        return output


def save_model(model, optimizer, PATH):
    if not os.path.exists(PATH):
        # Create a new directory because it does not exist
        os.makedirs(PATH)
    # save the model
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        PATH + "best_model.pth",
    )


def load_model(model, optimizer, PATH):
    checkpoint = torch.load(PATH + "best_model.pth")
    # open the model
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    return model, optimizer
