# -*- coding: utf-8 -*-

from pathlib import Path
from typing import List, Optional, Dict, Any
import re
import json
from tqdm import tqdm
import copy

RAWDATA_PATH = Path("./data/train.txt")
TARGET_DIR = Path("./data/cgec")


class DataProcessor:
  leading_dash_pattern = re.compile(r"^——(.*)")

  def get_gec_samples_from_str(
      self,
      text: str,
  ) -> Optional[List[Dict[str, Any]]]:
    _, _, source, *corrects = text.strip().split('\t')
    samples = []
    source = DataProcessor._remove_leading_dash(source)
    if len(source) > 0:
      common = dict()
      common["source"] = source
      if len(corrects) > 0:
        for cor in corrects:
          cor = DataProcessor._remove_leading_dash(cor)
          if len(cor) > 0:
            sample = copy.deepcopy(common)
            sample["correct"] = cor

            samples.append(sample)
        if len(samples) == 0:
          return None
      else:
        common["correct"] = source
        samples.append(common)
      return samples
    return None

  @classmethod
  def _remove_leading_dash(cls, text: str) -> str:
    matches = cls.leading_dash_pattern.findall(text)
    if len(matches) > 0:
      return matches[0]
    else:
      return text

  def get_trainset_valset_from_file(self, source_file: Path,
                                    save_dir: Path) -> None:
    trainset_path = save_dir / "train.txt"
    valset_path = save_dir / "val.txt"
    trainset = []
    valset = []
    val_num, i = 5000, 0

    source_data = open(source_file, "r").readlines()
    for line in tqdm(source_data):
      samples = self.get_gec_samples_from_str(line)
      if samples is not None:
        if i < val_num:
          if len(samples) == 1:
            valset.extend(samples)
            i += 1
          else:
            trainset.extend(samples)
        else:
          trainset.extend(samples)

    save_dir.mkdir(exist_ok=True)
    if len(trainset) > 0:
      save_dicts_to_file(trainset, trainset_path)
      save_dicts_to_file(valset, valset_path)


def save_dicts_to_file(dicts: List[Dict[str, Any]],
                       save_path: str,
                       encoding='utf-8') -> None:
  with open(save_path, "w+", encoding=encoding) as fout:
    for d in dicts:
      json.dump(d, fout, ensure_ascii=False)
      fout.write('\n')


def main():
  data_processor = DataProcessor()
  data_processor.get_trainset_valset_from_file(RAWDATA_PATH, TARGET_DIR)


if __name__ == '__main__':
  main()
