import cv2
import matplotlib.pylab as plt
import ipywidgets as widgets
import numpy as np

from IPython.display import display, clear_output
from ipywidgets import interactive

plt.style.use('default')


def show(img, size=3):
    plt.figure(figsize=(size, size))
    ax = plt.axes([0, 0, 1, 1], frameon=False)
    ax.set_axis_off()
    plt_im = plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.plot()


def show_cv2(img):
    cv2.imshow('lena', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def task2(lena):
    lena_gray = cv2.cvtColor(lena, cv2.COLOR_BGR2GRAY)

    lena_gray_linear = cv2.normalize(lena_gray, None, alpha=0, beta=256, norm_type=cv2.NORM_MINMAX)
    lena_gray_equalize = cv2.equalizeHist(lena_gray)

    diff = np.abs(lena_gray_linear.astype(int) - lena_gray_equalize.astype(int)).astype('uint8')

    imgs = np.hstack([lena_gray_linear, lena_gray_equalize, diff])
    show_cv2(imgs)


def task3(lena):
    channels = [lena[:, :, i] for i in range(3)]
    p = [0.05, 0.1, 0.15]
    q = p.copy()

    lenas = []
    for p, q in zip(p, q):
        lena_noise = np.dstack([salt_pepper(c, p, q) for c in channels])
        lenas.append(lena_noise)

    show_cv2(np.hstack(lenas))


def salt_pepper(src, p, q):
    src_ = src.copy()
    pixels = int(np.prod(src.shape))
    mat = np.random.choice(3, pixels, p=[p, q, 1 - p - q]).reshape(src_.shape).astype(int)

    zeros_mask = mat == 0
    ones_mask = mat == 1

    src_[zeros_mask] = 0
    src_[ones_mask] = 255

    return src_


def mse(im1, im2):
    e = im1.flatten() - im2.flatten()
    n = len(e)
    return 1 / n * e @ e


def remove_noise(img):
    img_ = img.copy()

    img_blurred = cv2.GaussianBlur(img_, (5, 5), 2)
    # img_blurred = cv2.medianBlur(img_, 7)
    img_diff = np.absolute(img_.astype(int) - img_blurred.astype(int))

    threshold = 60
    high_signal_mask = img_diff >= threshold

    img_res = img_blurred
    # img_res[~high_signal_mask] = img_[~high_signal_mask]

    return img_res


def task4():
    lena_noise = cv2.imread('data/lena_color_512-noise.tif')
    lena_target = cv2.imread('data/lena_color_512.tif')

    lena_cleared = remove_noise(lena_noise)

    print(f'Initial MSE: {mse(lena_noise, lena_target)}')
    print(f'After processing MSE: {mse(lena_cleared, lena_target)}')

    show_cv2(np.hstack([lena_target, lena_cleared]))


if __name__ == "__main__":
    lena = cv2.imread("data/lena.jpg")
    # show(lena)
    # task2(lena)
    # task3(lena)
    task4()


