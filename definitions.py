import os

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

IMGS_DIR = os.path.join(ROOT_DIR, "src/hw6/data/images")
TEST_PATH = os.path.join(ROOT_DIR, "src/hw6/data/test.dat")

N_WORKERS = 16
BATCH_SIZE = 64
K_NEIGHBOURS = 10
