# -*- coding: utf-8 -*-

import random

from typing import Dict, List, Tuple, Union
from pypinyin import lazy_pinyin


def is_chinese_token(token: str) -> bool:
  if ('\u4e00' <= token <= '\u9fa5'):
    return True
  return False


def get_token_pinyin(token: str) -> str:
  return lazy_pinyin(token)[0]


def get_chinese_tokens_from_map(
    token_id_map: Dict[str, int],
    return_dict=False) -> Union[Dict[str, int], List[int]]:
  if return_dict:
    chinese_token_id_map = dict()
  else:
    chinese_token_list = []

  for token, token_id in token_id_map.items():
    if len(token) == 1 and is_chinese_token(token):
      if return_dict:
        chinese_token_id_map[token] = token_id
      else:
        chinese_token_list.append(token)

  if return_dict:
    return chinese_token_id_map
  return chinese_token_list


def replace_chinese_token_by_chance(text: str,
                                    chinese_tokens,
                                    chance: int = 0.3) -> Tuple[str, int]:
  text_tokens = []
  replace_count = 0
  for i in range(len(text)):
    token = text[i]
    if is_chinese_token(
        token) and random.random() < chance:  # replace when i < chance
      token = random.choice(chinese_tokens)
      replace_count += 1
    text_tokens.append(token)

  return "".join(text_tokens), replace_count
