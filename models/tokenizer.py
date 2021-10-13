# -*- coding: utf-8 -*-

import jieba

from typing import List, Optional
from transformers.tokenization_utils import _is_whitespace, _is_control
from transformers.models.bert.tokenization_bert import BertTokenizer


def whitespace_tokenize(text):
  """Runs basic whitespace cleaning and splitting on a piece of text."""
  text = text.strip()
  if not text:
    return []
  # tokens = text.split()
  tokens = list(text)
  return tokens


class CgecTokenizer(BertTokenizer):

  def __init__(self,
               vocab_file,
               do_lower_case=False,
               do_basic_tokenize=True,
               never_split=None,
               unk_token="[UNK]",
               sep_token="[SEP]",
               pad_token="[PAD]",
               cls_token="[CLS]",
               mask_token="[MASK]",
               tokenize_chinese_chars=True,
               strip_accents=None,
               **kwargs):
    super().__init__(
        vocab_file=vocab_file,
        do_lower_case=do_lower_case,
        do_basic_tokenize=do_basic_tokenize,
        never_split=never_split,
        unk_token=unk_token,
        sep_token=sep_token,
        pad_token=pad_token,
        cls_token=cls_token,
        mask_token=mask_token,
        tokenize_chinese_chars=tokenize_chinese_chars,
        strip_accents=strip_accents,
        **kwargs,
    )

  def _tokenize(self, text, **kwargs):
    text = self._clean_text(text)
    if len(text) > 0:
      tokens = whitespace_tokenize(text)
      return tokens
    return ""

  def _clean_text(self, text):
    """Performs invalid character removal and whitespace cleanup on text."""
    output = []
    for char in text:
      cp = ord(char)
      if cp == 0 or cp == 0xFFFD or _is_control(char):
        continue
      if _is_whitespace(char):
        output.append(" ")
      else:
        output.append(char)
    return "".join(output)
