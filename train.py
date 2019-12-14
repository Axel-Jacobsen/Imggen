#! /usr/bin/env python3

import torch
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image

from model import Imggen


class TrainV0(object):

    # hyperparameters
    BATCH_SIZE = 64
    GAMMA = 0.99
    TARGET_UPDATE = 100
    REWARD_CONST = 1
    lr = 1e-1

    def __init__(self, model: Imggen, device: str, pth=None):
        self.model = model
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.steps = 0
        self.device = device
        self.static_noise = torch.randn(1, 16)
        if pth:
            self.model.load_state_dict(torch.load(pth))

    @staticmethod
    def make_img(byte_img):
        im_arr = byte_img.reshape(16,16,3).numpy()
        return Image.fromarray(im_arr, 'RGB')

    def train(self):
        ims = []
        for i in range(1000):
            noise = self.static_noise
            model_out = self.model(noise)

            # Give constant negative reward
            rewards = torch.FloatTensor([self.REWARD_CONST])

            # Calculate loss
            loss = torch.sum(torch.mul(model_out, rewards).mul(-1), -1)

            # Update network weights
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if i % 100 == 0:
                with torch.no_grad():
                    out = self.make_img(self.model(noise))
                ims.append(out)

        w = h = 10
        columns = 2
        rows = 5
        fig = plt.figure(figsize=(8,8))
        for i in range(1,columns*rows + 1):
            ax = fig.add_subplot(rows, columns, i)
            ax.set_title(f'{i * 100}')
            plt.imshow(ims[i - 1])
        plt.show()

    def gen_img(self, noise=None):
        if not noise: noise = torch.randn(1, 16)
        with torch.no_grad():
            out = self.model(noise)
        img = self.make_img(out)
        return img

    def plot_samples(self, noise=None):
        w = h = 10
        columns = rows = 4
        fig = plt.figure(figsize=(8, 8))
        for i in range(1, columns*rows +1):
            img = self.gen_img(noise)
            fig.add_subplot(rows, columns, i)
            plt.imshow(img)
        plt.show()


if __name__ == '__main__':
    device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

    mdl = Imggen(in_features=16, width=16, height=16)
    tr = TrainV0(mdl, device=device)
    tr.train()
    tr.plot_samples()

