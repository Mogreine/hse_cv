import os
from time import time

import numpy as np
import torch
import cv2
import multiprocessing as mp
import pandas as pd

from sklearn.cluster import KMeans

from src.hw6.mine import mean_average_precision
from definitions import ROOT_DIR, IMGS_DIR, TEST_PATH, N_WORKERS, BATCH_SIZE, K_NEIGHBOURS, PRETRAINED_EMBEDDINGS_PATH


class Gist:
    def __init__(self, ksize=31, depth=5, step=3, number_of_orientation=6, window_size=4):
        self.lambd_array = np.array([step * (i + 1) for i in range(depth)], dtype="float")
        self.theta_array = np.arange(0, np.pi, np.pi / number_of_orientation)

        self.h, self.w = depth, number_of_orientation
        self.window_size = window_size
        self.filters = []

        for lambd in self.lambd_array:
            sigma = 1.5 * lambd / 5
            for theta in self.theta_array:
                kern = cv2.getGaborKernel((ksize, ksize), sigma, theta, lambd, gamma=1, psi=0)
                kern /= np.sqrt((kern * kern).sum())
                self.filters.append(kern.copy())
        self.filters = np.array(self.filters)

    def __call__(self, image):
        return self.get_gist_descriptor(image)

    def get_gist_descriptor(self, image):
        hight, width, chanel = image.shape

        res = np.zeros((self.h * hight, self.w * width, chanel), dtype="float")
        res1 = np.zeros((self.h * self.window_size, self.w * self.window_size, chanel), dtype="float")

        descriptor = []

        for i in range(self.h):
            for j in range(self.w):
                temp = cv2.filter2D(image, -1, self.filters[self.w * i + j])
                res[i * hight : (i + 1) * hight, j * width : (j + 1) * width] = cv2.normalize(
                    temp, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F
                )
                for y in range(self.window_size):
                    for x in range(self.window_size):
                        res1[i * self.window_size + y, j * self.window_size + x] = np.mean(
                            res[
                                i * hight
                                + int(y * hight / self.window_size) : i * hight
                                + int((y + 1) * hight / self.window_size),
                                j * width
                                + int(x * width / self.window_size) : j * width
                                + int((x + 1) * width / self.window_size),
                            ],
                            axis=(0, 1),
                        )
                descriptor.extend(
                    res1[
                        i * self.window_size : (i + 1) * self.window_size,
                        j * self.window_size : (j + 1) * self.window_size,
                    ].flatten()
                )
        return np.array(descriptor)


class LshCode:
    def __init__(self, centroid, descriptors, code_length):
        self.centroid = centroid.copy()
        points = (descriptors - centroid).copy()
        self.code_length = code_length
        line_length = points.shape[1]
        self.norm_array = np.random.uniform(-100, 100, (self.code_length, line_length))
        self.norm_array /= np.linalg.norm(self.norm_array, axis=1).reshape(-1, 1)
        self.d_array = np.random.normal(0, np.median(np.std(points, 0)), self.code_length)
        self.lsh_codes = np.zeros((self.code_length, len(points)))
        for i in range(self.code_length):
            self.lsh_codes[i] = points @ self.norm_array[i] + self.d_array[i] > 0
        self.lsh_codes = self.lsh_codes.T

    def get_norm(self):
        return self.norm_array

    def get_d(self):
        return self.d_array

    def get_lsh_codes(self):
        return self.lsh_codes

    def create_lsh_code(self, point):
        p = (point - self.centroid).copy()
        lsh_code = np.zeros(self.code_length)
        for i in range(self.code_length):
            lsh_code[i] = p.dot(self.norm_array[i]) + self.d_array[i] > 0
        return lsh_code


class Searcher:
    def __init__(self, n_clusters):
        self.encoder = Gist()
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        self.labels = None
        self.lsh_codes = []
        self.lsh_f = []

    def _get_embeddings(self, images: np.ndarray):
        return np.array([self.encoder(im) for im in images])

    def fit(self, images, lsh_code_len):
        self.lsh_code_len = lsh_code_len
        self.im_desc = self._get_embeddings(images)

        self.kmeans.fit(self.im_desc)
        self.labels = self.kmeans.labels_
        self.centroids = self.kmeans.cluster_centers_

        for i, centroid in enumerate(self.centroids):
            lsh = LshCode(centroid, self.im_desc[self.labels == i], lsh_code_len)
            self.lsh_codes.append(lsh.get_lsh_codes())
            self.lsh_f.append(lsh)

    def retrieve(self, images, n_nei: int = 10):
        im_encoded = self._get_embeddings(images)
        res = []
        for im_emb in im_encoded:
            dist2clusters = np.linalg.norm(self.centroids - im_emb, axis=1).flatten()

            closest_clusters = np.argsort(dist2clusters)[:int(0.1*len(self.centroids))]

            ims2consider = np.zeros(len(self.labels), dtype=np.bool)

            for cl in closest_clusters:
                lsh_arr = self.lsh_codes[cl]
                im_lsh = self.lsh_f[cl].create_lsh_code(im_emb)

                ims2consider[self.labels == cl] = np.sum(np.abs(lsh_arr - im_lsh), axis=1) < int(0.5 * self.lsh_code_len)


            dist2ims = np.ones_like(ims2consider) * 1e9
            dist2ims[ims2consider] = np.linalg.norm(self.im_desc[ims2consider] - im_emb, axis=1)

            res.append(np.argsort(dist2ims)[:n_nei])

        return res


def read_and_convert(name):
    return cv2.cvtColor(cv2.imread(name), cv2.COLOR_BGR2RGB)


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

    searcher = Searcher(25)

    print("Fitting model...")
    start = time()
    searcher.fit(train, 10)
    print(f"Finished fitting, elapsed time: {(time() - start) / 60: .2f} min")

    print("Validating...")
    start = time()
    ids = searcher.retrieve(test, K_NEIGHBOURS)
    print(f"Finished validating, elapsed time: {(time() - start) / len(test): .4f} sec for a single query")

    preds = np.array([train_ids[pred] for pred in ids])
    map = mean_average_precision(preds, test_ids)
    print(f"MAP: {map: .4f}")

    print("Done!")
