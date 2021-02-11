import cv2
import matplotlib.pylab as plt
import numpy as np
from itertools import permutations


def show_img(img):
    cv2.imshow('lena', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def task3():
    lena = cv2.imread('lena.png')
    show_img(lena)
    return lena


def task4(lena):
    n, m, _ = lena.shape
    tmp = lena[:n//2, :m//2].copy()
    lena[:n // 2, :m // 2] = lena[n//2:, m//2:]
    lena[n // 2:, m // 2:] = tmp

    show_img(lena)


def task5(lena):
    lena_gray_my = np.mean(lena, axis=2).astype('int')
    return lena_gray_my


def task6(lena, lena_gray_my):
    lena_gray_cvt = cv2.cvtColor(lena, cv2.COLOR_BGR2GRAY).astype('int')
    lena_diff = np.abs(lena_gray_my - lena_gray_cvt).astype('uint8')

    show_img(lena_diff)


def task7(lena):
    lena_hsv = cv2.cvtColor(lena, cv2.COLOR_BGR2HSV)
    lena_hsv[:, :, 1] = 255
    lena_hsv[:, :, 2] = 255
    lena_rgb = cv2.cvtColor(lena_hsv, cv2.COLOR_HSV2BGR)

    show_img(lena_rgb)


def task8(lena):
    lena_hsv = cv2.cvtColor(lena, cv2.COLOR_BGR2HSV)
    lena_hsv[:, :, 2] = 255 - lena_hsv[:, :, 2]
    lena_rgb = cv2.cvtColor(lena_hsv, cv2.COLOR_HSV2BGR)

    show_img(lena_rgb)


def task9(lena):
    b, g, r = lena[:, :, 0], lena[:, :, 1], lena[:, :, 2]
    perms = list(permutations((b, g, r), 3))
    imgs = list(map(lambda c_arr: np.stack(c_arr, axis=2), perms))
    imgs = [np.concatenate([imgs[i], imgs[i + 1]], axis=1) for i in range(0, 6, 2)]
    imgs = np.concatenate(imgs, axis=0)

    show_img(imgs)


if __name__ == "__main__":
    lena = task3()
    task4(lena.copy())
    gray = task5(lena)
    task6(lena, gray)
    task7(lena)
    task8(lena)
    task9(lena)
