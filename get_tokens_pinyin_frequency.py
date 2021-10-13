# -*- coding: utf-8 -*-

from tqdm import tqdm
import json

from collections import Counter, defaultdict
from models.utils import is_chinese_token, get_token_pinyin

from typing import Dict, List, Optional, Tuple


def get_frequency(
    file_path: str,
    need_pinyin_freq: bool = False
) -> Tuple[Tuple[List[str], List[int]], Optional[Dict[str, Tuple[List[str],
                                                                 List[int]]]]]:
  lines = open(file_path, 'r').readlines()
  token_counter = Counter()
  if need_pinyin_freq:
    pinyin_token_dict = defaultdict(Counter)
  for line in tqdm(lines):
    _, num_targets, source, *targets = line.split('\t')
    if num_targets == 0:
      targets = [source]
    for target in targets:
      chinese = [token for token in target if is_chinese_token(token)]
      token_counter.update(chinese)
      if need_pinyin_freq:
        for token in chinese:
          token_pinyin = get_token_pinyin(token)
          token_pinyin_counter = pinyin_token_dict[token_pinyin]
          token_pinyin_counter.update([token])
  token_total = sum(token_counter.values())
  tokens = [token for token in token_counter.keys()]
  tokens_freq = [value / token_total for value in token_counter.values()]
  pingyin_tokens_freq = dict()
  if need_pinyin_freq:
    for pinyin, pinyin_counter in pinyin_token_dict.items():
      pinyin_tokens = [token for token in pinyin_counter.keys()]
      pinyin_total = sum(pinyin_counter.values())
      pinyin_freq = [value / pinyin_total for value in pinyin_counter.values()]
      pingyin_tokens_freq[pinyin] = (pinyin_tokens, pinyin_freq)
    return {"tokens": tokens, "tokens_freq": tokens_freq}, pingyin_tokens_freq
  return {"tokens": tokens, "tokens_freq": tokens_freq}


def main():
  tokens_freq_dict, pinyin_freq_dict = get_frequency(
      "./data/train.txt",
      need_pinyin_freq=True,
  )
  json.dump(
      tokens_freq_dict,
      open("./resources/tokens_freq.json", "w"),
      ensure_ascii=False,
      indent=4,
  )
  json.dump(
      pinyin_freq_dict,
      open("./resources/pinyin_freq.json", "w"),
      ensure_ascii=False,
      indent=4,
  )


if __name__ == '__main__':
  main()
