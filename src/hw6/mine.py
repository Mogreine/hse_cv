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

from definitions import ROOT_DIR, IMGS_DIR, TEST_PATH, N_WORKERS, BATCH_SIZE, K_NEIGHBOURS


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
    train_img_ids = np.array([ids for ids in range(len(images)) if ids not in test_img_ids])

    return images[train_img_ids], train_img_ids, images[test_img_ids], test_img_ids


class ImageSearcher:
    def __init__(
        self,
        device: torch.device = torch.device("cpu"),
        n_nei: int = 5,
        algorithm: str = "auto",
        leaf_size: int = 100,
        n_jobs: int = -1,
    ):
        self.cluster_model = NearestNeighbors(
            n_neighbors=n_nei, algorithm=algorithm, leaf_size=leaf_size, n_jobs=n_jobs
        )

        self.emb_model = EfficientNet.from_pretrained("efficientnet-b0")
        self.emb_model.to(device)
        self.emb_model.eval()

        for p in self.emb_model.parameters():
            p.requires_grad = False

        self.tfms = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

        self.images = None
        self.device = device

    def prep_image(self, images):
        tmp = [self.tfms(im.convert("RGB")).unsqueeze(0) for im in images]
        # tmp = []
        # for im in images:
        #     im = im.convert("RGB")
        #     tmp.append(self.tfms(im).unsqueeze(0))
        return torch.cat(tmp, dim=0)

    def get_embeddings(self, images: torch.Tensor, batch_size: int = 32):
        steps = len(images) // batch_size + (len(images) % batch_size != 0)
        res = []
        for i in tqdm(range(steps)):
            batch = images[i * batch_size : (i + 1) * batch_size].to(self.device)

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

    def retrieve(self, images, n_nei: int, batch_size: int = 32):
        assert self.images is not None, "First fit the model"

        im_transformed = self.prep_image(images)
        im_embedding = self.get_embeddings(im_transformed, batch_size).numpy()

        nei_ids = self.cluster_model.kneighbors(im_embedding, n_neighbors=n_nei, return_distance=False)

        similar_ims = [np.expand_dims(self.images[ids], axis=0) for ids in nei_ids]
        similar_ims = np.stack(similar_ims, axis=0)

        return nei_ids, similar_ims


def mean_average_precision(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """Calculate mean average precision (k == 1..n_preds)

    @param y_pred: [batch size; n preds] - predicted clusters
    @param y_true: [batch size; 1] - ground truth clusters
    @return: float - MAP@n_preds
    """
    if len(y_true.shape) == 1:
        y_true = y_true.reshape(-1, 1)

    correct_pred = y_pred == y_true

    # [batch size; n preds]
    true_positives = np.cumsum(correct_pred, axis=-1)
    precision_k = true_positives / np.arange(1, true_positives.shape[1] + 1).reshape(1, -1)

    precision_k[~correct_pred] = 0
    n_correct = correct_pred.sum(-1)
    n_correct[n_correct == 0] = 1  # to avoid zero division (numerator is zero, thus 0/1=0)
    average_precision = precision_k.sum(-1) / n_correct

    return average_precision.mean()


if __name__ == "__main__":
    print("Reading imgs...")
    train, train_ids, test, test_ids = read_imgs()
    print("Done reading")

    # ids to classes
    train_ids = train_ids // 100
    test_ids = test_ids // 100

    train_sz = None
    test_sz = None

    train = train[:train_sz]
    train_ids = train_ids[:train_sz]
    test = test[:test_sz]
    test_ids = test_ids[:test_sz]

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    searcher = ImageSearcher(device)

    print("Fitting model...")
    start = time()
    searcher.fit(train, BATCH_SIZE)
    print(f"Finished fitting, elapsed time: {(time() - start) / 60: .2f} min")

    print("Validating...")
    start = time()
    ids, similar_ims = searcher.retrieve(test, K_NEIGHBOURS, BATCH_SIZE)
    print(f"Finished validating, elapsed time: {(time() - start) / len(test): .4f} sec for a single query")

    preds = np.array([train_ids[pred] for pred in ids])
    map = mean_average_precision(preds, test_ids)
    print(f"MAP: {map: .4f}")

    print("Done!")
