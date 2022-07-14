from transformers import BertTokenizer, AutoTokenizer, RobertaTokenizer
import click
import torch


class ProteinTokenizer:
    def __init__(self, vocab=None):
        if vocab is None :
            vocab = 'data/external/protein_vocab.txt'
        self.tokenizer = BertTokenizer(vocab, do_lower_case=False
        )

    @property
    def vocab_size(self):
        return self.tokenizer.vocab_size

    def encode(self, sequence):
        return self.tokenizer.encode(" ".join(list(sequence)), add_special_tokens=True)

    def tokenize(self, sequence):
        return self.tokenizer.tokenize(" ".join(list(sequence)), add_special_tokens=True)

    def batch_encode_plus(self, sequences, max_length=None, **kwargs):

        ids = [self.encode(smi) for smi in sequences]
        if max_length:
            ids = ids[:max_length]
        max_len = max([len(x) for x in ids])
        input_ids, masks = [], []
        for item in ids:
            diff = max_len - len(item)
            input_ids.append(item + [self.tokenizer.pad_token_id] * diff)
            masks.append(len(item) * [1] + [0] * diff)

        return {'input_ids': input_ids,
                'attention_mask': masks}

    def batch_decode(self, batch):
        return self.tokenizer.batch_decode(batch)


class ProteinBPETokenizer(RobertaTokenizer):
    def __init__(self, path=None):
        if path is None:
            path = 'data/external/paccmann'
        super().__init__(vocab_file=f"{path}/vocab.json", merges_file=f"{path}/merges.txt")



