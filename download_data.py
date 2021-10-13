# -*- coding: utf-8 -*-

import requests
import os
import logging
import tarfile
import zipfile
import shutil

from pathlib import Path
from tqdm import tqdm

TRAIN_DATA_URL = "http://tcci.ccf.org.cn/conference/2018/dldoc/trainingdata02.tar.gz"
TEST_DATA_URL = "http://tcci.ccf.org.cn/conference/2018/dldoc/tasktestdata02.zip"

logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)


def download(url, file_name):
  # open in binary mode
  with open(file_name, "wb") as file:
    # get request
    response = requests.get(url, stream=True)
    total_size_in_bytes = int(response.headers.get('content-length', 0))
    block_size = 1024  #1 Kibibyte
    progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True)
    for data in response.iter_content(block_size):
      progress_bar.update(len(data))
      file.write(data)
    progress_bar.close()
    if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
      print("ERROR, something went wrong")


def main():
  Path("data").mkdir(exist_ok=True)
  if not Path("data/train.txt").exists():
    logger.info("downloading training data:")
    download(TRAIN_DATA_URL, Path("data/trainingdata02.tar.gz"))
    tar = tarfile.open(Path("data/trainingdata02.tar.gz"), "r:gz")
    tar.extractall("data")
    shutil.move("data/NLPCC2018_GEC_TrainingData/data.train", "data/train.txt")
    tar.close()
    shutil.rmtree("data/NLPCC2018_GEC_TrainingData")
    os.remove("data/trainingdata02.tar.gz")

  if not Path("data/test.txt").exists():
    logger.info("downloading test data:")
    download(TEST_DATA_URL, Path("data/tasktestdata02.zip"))
    zip = zipfile.ZipFile("data/tasktestdata02.zip")
    zip.extractall("data")
    shutil.move("data/TestData_Task2/source.txt", "data/test.txt")
    shutil.rmtree("data/TestData_Task2")
    os.remove("data/tasktestdata02.zip")

  pass


if __name__ == '__main__':
  main()
