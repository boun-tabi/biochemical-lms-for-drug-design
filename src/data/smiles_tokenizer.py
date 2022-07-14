
import collections
import os
import re
import click
import logging
from typing import List
from transformers import BertTokenizer, RobertaTokenizer
from logging import getLogger


logger = getLogger(__name__)
"""
SMI_REGEX_PATTERN: str
    SMILES regex pattern for tokenization. Designed by Schwaller et. al.

References

.. [1]  Philippe Schwaller, Teodoro Laino, Théophile Gaudin, Peter Bolgar, Christopher A. Hunter, Costas Bekas, and Alpha A. Lee
        ACS Central Science 2019 5 (9): Molecular Transformer: A Model for Uncertainty-Calibrated Chemical Reaction Prediction
        1572-1583 DOI: 10.1021/acscentsci.9b00576

"""

SMI_REGEX_PATTERN = r"""(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|
#|-|\+|\\|\/|:|~|@|\?|>>?|\*|\$|\%[0-9]{2}|[0-9])"""

# add vocab_file dict
VOCAB_FILES_NAMES = {"vocab_file": "vocab.txt"}


class SmilesTokenizer(BertTokenizer):
    """
      SMILESTokenizer implementation from DeepChem
    """
    vocab_files_names = VOCAB_FILES_NAMES

    def __init__(
            self,
            vocab_file=None,
            # unk_token="[UNK]",
            # sep_token="[SEP]",
            # pad_token="[PAD]",
            # cls_token="[CLS]",
            # mask_token="[MASK]",
            **kwargs):

        if vocab_file is None:
            vocab_file = 'data/external/smiles_vocab.txt'
        super().__init__(vocab_file, **kwargs)

        if not os.path.isfile(vocab_file):
            raise ValueError(
                "Can't find a vocab file at path '{}'.".format(vocab_file))
        self.vocab = load_vocab(vocab_file)
        self.highest_unused_index = max(
            [i for i, v in enumerate(self.vocab.keys()) if v.startswith("[unused")])
        self.ids_to_tokens = collections.OrderedDict(
            [(ids, tok) for tok, ids in self.vocab.items()])
        self.basic_tokenizer = BasicSmilesTokenizer()
        # self.init_kwargs["max_len"] = self.max_len

    @property
    def vocab_size(self):
        return len(self.vocab)

    @property
    def vocab_list(self):
        return list(self.vocab.keys())

    @property
    def bos_token_id(self):
        return self.cls_token_id


    @property
    def eos_token_id(self):
        return self.sep_token_id

    def _tokenize(self, text: str):
        split_tokens = [token for token in self.basic_tokenizer.tokenize(text)]
        return split_tokens

    def _convert_token_to_id(self, token):
        return self.vocab.get(token, self.vocab.get(self.unk_token))

    def _convert_id_to_token(self, index):
        return self.ids_to_tokens.get(index, self.unk_token)

    def convert_tokens_to_string(self, tokens: List[str]):
        out_string: str = " ".join(tokens).replace(" ##", "").strip()
        return out_string

    def add_special_tokens_ids_single_sequence(self, token_ids: List[int]):
        return [self.cls_token_id] + token_ids + [self.sep_token_id]

    def add_special_tokens_single_sequence(self, tokens: List[str]):
        return [self.cls_token] + tokens + [self.sep_token]

    def add_special_tokens_ids_sequence_pair(self, token_ids_0: List[int],
                                             token_ids_1: List[int]) -> List[int]:
        sep = [self.sep_token_id]
        cls = [self.cls_token_id]

        return cls + token_ids_0 + sep + token_ids_1 + sep

    def add_padding_tokens(self,
                           token_ids: List[int],
                           length: int,
                           right: bool = True) -> List[int]:
        padding = [self.pad_token_id] * (length - len(token_ids))

        if right:
            return token_ids + padding
        else:
            return padding + token_ids

    def save_vocabulary(
            self, vocab_path: str, filename_prefix=None
    ):  # -> tuple[str]: doctest issue raised with this return type annotation
        index = 0
        if os.path.isdir(vocab_path):
            vocab_file = os.path.join(vocab_path, VOCAB_FILES_NAMES["vocab_file"])
        else:
            vocab_file = vocab_path
        with open(vocab_file, "w", encoding="utf-8") as writer:
            for token, token_index in sorted(
                    self.vocab.items(), key=lambda kv: kv[1]):
                if index != token_index:
                    logger.warning(
                        "Saving vocabulary to {}: vocabulary indices are not consecutive."
                        " Please check that the vocabulary is not corrupted!".format(
                            vocab_file))
                    index = token_index
                writer.write(token + "\n")
                index += 1
        return (vocab_file,)

    def decode(self, output_ids, **kwargs):
        decoded_with_spaces = super().decode(output_ids, **kwargs)
        return ''.join(decoded_with_spaces.split())

    def batch_decode(self, batch, **kwargs):
        decoded_with_spaces = super().batch_decode(batch, **kwargs)
        return [''.join(item.split()) for item in decoded_with_spaces]


class BasicSmilesTokenizer(object):
    def __init__(self, regex_pattern: str = SMI_REGEX_PATTERN):
        self.regex_pattern = regex_pattern
        self.regex = re.compile(self.regex_pattern)

    def tokenize(self, text):
        tokens = [token for token in self.regex.findall(text)]
        return tokens


def load_vocab(vocab_file):
    """Loads a vocabulary file into a dictionary."""
    vocab = collections.OrderedDict()
    with open(vocab_file, "r", encoding="utf-8") as reader:
        tokens = reader.readlines()
    for index, token in enumerate(tokens):
        token = token.rstrip("\n")
        vocab[token] = index
    return vocab


class SmilesBPETokenizer(RobertaTokenizer):

    def __init__(self, path=None):
        if path is None:
            path = 'data/external/chemberta'
        super().__init__(vocab_file=f"{path}/vocab.json", merges_file=f"{path}/merges.txt")
