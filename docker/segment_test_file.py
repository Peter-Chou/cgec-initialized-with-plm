# -*- coding: utf-8 -*-

import sys
import csv
import os

from pathlib import Path
import argparse

sys.path.append(os.fspath(Path("/opt")))

from pkunlp import Segmentor


def save_results_to_file(results, save_path, delimiter='\t', encoding="utf-8"):
  """save results to file

  Args:
    results (Iterable): A iterable of results
    save_path (str): the path of file which results save to
    delimiter (str): A string indicates items are separated by delimiter char
    encoding (str): encoding for saved file
  """
  with open(save_path, 'w', newline='', encoding=encoding) as fout:
    writer = csv.writer(fout, delimiter=delimiter)
    for result in results:
      writer.writerow(result)


def segment_test_file(result_path, segmented_file, segmentor):
  segmented_results = []
  with open(result_path, 'r', encoding="utf-8") as fin:
    for line in fin:
      result_str = line.strip()
      segmented_list = segmentor.seg_string(result_str)
      segmented_results.append(segmented_list)

  save_results_to_file(segmented_results, segmented_file, delimiter=" ")


def main():

  parser = argparse.ArgumentParser()

  parser.add_argument("-i",
                      "--input",
                      type=str,
                      required=True,
                      help="text needs to be segmented")

  args = parser.parse_args()
  segmentor = Segmentor("/opt/feature/segment.feat", "/opt/feature/segment.dic")
  output_file = os.path.splitext(args.input)[0] + "_segmented.txt"
  segment_test_file(args.input, output_file, segmentor)


if __name__ == '__main__':
  main()
