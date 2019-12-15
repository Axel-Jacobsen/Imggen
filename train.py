#! /usr/bin/env python3

import math
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
    lr = 1e-3

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
        loops = 100
        for i in range(10 * loops):
            model_out = self.model(self.static_noise)

            # Give constant negative reward
            reward = self.REWARD_CONST if np.random.rand() < 0.75 else -1 * self.REWARD_CONST
            rewards = torch.FloatTensor([reward])

            # Calculate loss
            loss = torch.sum(torch.mul(model_out, rewards).mul(-1), -1)

            # Update network weights
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if i % loops == 0:
                with torch.no_grad():
                    out = self.make_img(self.model(self.static_noise))
                ims.append(out)

        self.plot_samples(ims)

    def gen_img(self, noise=None):
        if not noise: noise = torch.randn(1, 16)
        with torch.no_grad():
            out = self.model(noise)
        img = self.make_img(out)
        return img

    def plot_samples(self, ims, noise=None):
        h = 2
        w = math.ceil(len(ims) / 2)
        fig = plt.figure(figsize=(6, 6))
        i = 1
        for img in ims:
            ax = fig.add_subplot(h, w, i)
            ax.axis('off')
            ax.set_title(f'{i}')
            plt.imshow(img)
            i += 1
        plt.show()


if __name__ == '__main__':
    device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

    mdl = Imggen(in_features=16, width=16, height=16)
    tr = TrainV0(mdl, device=device)
    tr.train()

