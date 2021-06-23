import os

import numpy as np
import pandas as pd
import cv2
import glob
import time
import pickle
import torch
import multiprocessing as mp
from sklearn.cluster import KMeans
import matplotlib.pylab as plt
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

from copy import deepcopy
from efficientnet_pytorch import EfficientNet
from sklearn.neighbors import NearestNeighbors
from time import time

from definitions import ROOT_DIR, IMGS_DIR, TEST_PATH, N_WORKERS


def read_and_convert(name):
    return Image.open(name)


def read_imgs():
    p = mp.Pool(N_WORKERS)
    img_names = np.sort(
        [file_name for file_name in os.listdir(IMGS_DIR) if os.path.isfile(os.path.join(IMGS_DIR, file_name))]
    )
    img_names = [os.path.join(IMGS_DIR, file_name) for file_name in img_names]
    images = p.map(read_and_convert, img_names)
    images = np.array(images)
    p.close()

    df_test = pd.read_csv(TEST_PATH, header=None, names=["labels"])
    test_img_ids = np.array(list(map(lambda x: int(x[:-5]), df_test.labels.values)), dtype="int")
    train_img_ids = [ids for ids in range(len(images)) if ids not in test_img_ids]

    return images[train_img_ids], images[test_img_ids]


class ImageSearcher:
    def __init__(self, device: str = "cpu", n_nei: int = 5, algorithm: str = "auto", leaf_size: int = 100, n_jobs: int = -1):
        self.cluster_model = NearestNeighbors(n_neighbors=n_nei, algorithm=algorithm, leaf_size=leaf_size, n_jobs=n_jobs)

        self.emb_model = EfficientNet.from_pretrained('efficientnet-b0')
        self.emb_model.to(device)
        self.emb_model.eval()

        for p in self.emb_model.parameters():
            p.requires_grad = False

        self.tfms = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]), ])

        self.images = None
        self.device = device

    def prep_image(self, images):
        return torch.cat([self.tfms(im).unsqueeze(0) for im in images], dim=0)

    def get_embeddings(self, images: torch.Tensor, batch_size: int = 32):
        steps = len(images) // batch_size + (len(images) % batch_size != 0)
        res = []
        for i in tqdm(range(steps)):
            batch = images[i * batch_size: (i + 1) * batch_size].to(self.device)

            res.append(self.emb_model.extract_features(batch).flatten(start_dim=1))

        res = torch.cat(res, dim=0)

        return res.cpu()

    def fit(self, images, batch_size: int = 32):
        # assert len(images.shape) == 4, "Wrong shapes"

        self.images = deepcopy(images)

        im_transformed = self.prep_image(images)
        im_embedding = self.get_embeddings(im_transformed, batch_size).numpy()

        print("Finished extracting features")

        self.cluster_model.fit(im_embedding)

    def find_closest(self, images, n_nei: int, batch_size: int = 32):
        assert self.images is not None, "First fit the model"

        im_transformed = self.prep_image(images)
        im_embedding = self.get_embeddings(im_transformed, batch_size).numpy()

        nei_ids = self.cluster_model.kneighbors(im_embedding, n_neighbors=n_nei, return_distance=False)

        res = [np.expand_dims(self.images[ids], axis=0) for ids in nei_ids]
        res = np.stack(res, axis=0)

        return res


if __name__ == "__main__":
    print("Reading imgs...")
    train, test = read_imgs()
    print("Done reading")

    searcher = ImageSearcher()

    print("Fitting model...")
    start = time()
    searcher.fit(train, 256)
    print(f"Finished fitting, elapsed time: {(time() - start) / 60} min")

    print("Validating...")
    start = time()
    searcher.find_closest(test, 5, 32)
    print(f"Finished validating, elapsed time: {(time() - start) / 60 / len(test)} min for a single query")

    print("Done!")
