# -*- coding: utf-8 -*-

import random
import json

from models.utils import is_chinese_token, get_token_pinyin


class ErrorMaker:

  def __init__(
      self,
      token_freq_file,
      pinyin_freq_file,
      text_mask_ratio=0.5,
      token_mask_ratio=0.3,
  ):
    self.text_mask_ratio = text_mask_ratio
    self.token_mask_ratio = token_mask_ratio
    self.tokens, self.tokens_freq = ErrorMaker.get_tokens_freq(token_freq_file)
    self.pinyin_dict = ErrorMaker.get_pinyin_freq(pinyin_freq_file)
    self.strategy_list = [
        self.replace_token_by_chance,
        self.replace_token_by_weight,
        self.replace_token_by_pinyin_weight,
        None,  # unchange stratey
    ]

  def dynamic_mask(self, text):
    is_changed = False
    if random.random() <= self.text_mask_ratio:
      text_tokens = []
      for i in range(len(text)):
        token = text[i]
        if is_chinese_token(token) and random.random(
        ) < self.token_mask_ratio:  # replace when i < chance
          mask_strategy = random.choice(self.strategy_list)
          if mask_strategy is not None:
            token = mask_strategy(token)
            is_changed = True
        text_tokens.append(token)
      return is_changed, "".join(text_tokens)

    return is_changed, text

  def replace_token_by_chance(self, token):
    return random.choice(self.tokens)

  def replace_token_by_weight(self, token):
    return self._replace_token_by_weight(self.tokens, self.tokens_freq)

  def _replace_token_by_weight(self, tokens, tokens_freq):
    return random.choices(tokens, tokens_freq)[0]

  def replace_token_by_pinyin_weight(self, token):
    token_pinyin = get_token_pinyin(token)
    freq_tuple = self.pinyin_dict.get(token_pinyin, None)
    if freq_tuple is not None:
      pinyin_tokens, pinyin_tokens_freq = freq_tuple[0], freq_tuple[1]
      return self._replace_token_by_weight(pinyin_tokens, pinyin_tokens_freq)
    return token

  @staticmethod
  def get_tokens_freq(file_path: str):
    with open(file_path, 'r') as fin:
      token_dict = json.load(fin)
      return token_dict["tokens"], token_dict["tokens_freq"]

  @staticmethod
  def get_pinyin_freq(file_path: str):
    with open(file_path, 'r') as fin:
      return json.load(fin)
