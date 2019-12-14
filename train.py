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
    lr = 1e-3

    def __init__(self, model: Imggen, device: str, pth=None):
        self.model = model
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.steps = 0
        self.device = device
        if pth:
            self.model.load_state_dict(torch.load(pth))

    @staticmethod
    def make_img(byte_img):
        im_arr = byte_img.reshape(16,16,3).numpy()
        return Image.fromarray(im_arr, 'RGB')

    def train(self):
        pass

    def gen_img(self):
        noise = torch.randn(1, 16)
        with torch.no_grad():
            out = self.model(noise)
        img = self.make_img(out)
        return img

    def plot_samples(self):
        w=10
        h=10
        fig=plt.figure(figsize=(8, 8))
        columns = 4
        rows = 5
        for i in range(1, columns*rows +1):
            img = train.gen_img()
            fig.add_subplot(rows, columns, i)
            plt.imshow(img)
        plt.show()


if __name__ == '__main__':
    device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

    mdl = Imggen(in_features=16, width=16, height=16)
    train = TrainV0(mdl, device=device)
    train.plot_samples()

