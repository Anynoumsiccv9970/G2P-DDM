import os
from collections import defaultdict
import torch
import re

SPACE_NORMALIZER = re.compile(r"\s+")


def tokenize_line(line):
    line = SPACE_NORMALIZER.sub(" ", line)
    line = line.strip()
    return line.split()

class Dictionary(object):
    """A mapping from symbols to consecutive integers"""

    def __init__(
        self,
        pad='<pad>',
        blank='<blank>',
        eos='</s>',
        unk='<unk>',
        bos='<s>',
        mask='<mask>',
        extra_special_symbols=None,
    ):
        self.unk_word, self.pad_word, self.eos_word, self.mask_word = unk, pad, eos, mask
        self.symbols = []
        self.count = []
        self.indices = {}
        self.blank_index = self.add_symbol(blank)
        self.bos_index = self.add_symbol(bos)
        self.pad_index = self.add_symbol(pad)
        self.eos_index = self.add_symbol(eos)
        self.unk_index = self.add_symbol(unk)
        self.mask_index = self.add_symbol(mask)
        if extra_special_symbols:
            for s in extra_special_symbols:
                self.add_symbol(s)
        self.nspecial = len(self.symbols)


    def __eq__(self, other):
        return self.indices == other.indices

    def __getitem__(self, idx):
        if idx < len(self.symbols):
            return self.symbols[idx]
        return self.unk_word
    
    def __len__(self):
        """Returns the number of symbols in the dictionary"""
        return len(self.symbols)

    def add_symbol(self, word, n=1):
        """Adds a word to the dictionary"""
        if word in self.indices:
            idx = self.indices[word]
            self.count[idx] = self.count[idx] + n
            return idx
        else:
            idx = len(self.symbols)
            self.indices[word] = idx
            self.symbols.append(word)
            self.count.append(n)
            return idx

    def blank(self):
        """Helper to get index of beginning-of-sentence symbol"""
        return self.blank_index

    def bos(self):
        """Helper to get index of beginning-of-sentence symbol"""
        return self.bos_index

    def pad(self):
        """Helper to get index of pad symbol"""
        return self.pad_index

    def eos(self):
        """Helper to get index of end-of-sentence symbol"""
        return self.eos_index

    def unk(self):
        """Helper to get index of unk symbol"""
        return self.unk_index

    def mask(self):
        """Helper to get index of mask symbol"""
        return self.mask_index

    @classmethod
    def load(cls, f, ignore_utf_errors=False):
        """Loads the dictionary from a text file with the format:
        ```
        <symbol0> <count0>
        <symbol1> <count1>
        ...
        ```
        """
        d = cls()
        d.add_from_file(f, ignore_utf_errors)
        return d

    def add_from_file(self, f, ignore_utf_errors=False):
        """
        Loads a pre-existing dictionary from a text file and adds its symbols
        to this instance.
        """
        if isinstance(f, str):
            try:
                if not ignore_utf_errors:
                    with open(f, 'r', encoding='utf-8') as fd:
                        self.add_from_file(fd)
                else:
                    with open(f, 'r', encoding='utf-8', errors='ignore') as fd:
                        self.add_from_file(fd)
            except FileNotFoundError as fnfe:
                raise fnfe
            except UnicodeError:
                raise Exception("Incorrect encoding detected in {}, please "
                                "rebuild the dataset".format(f))
            return

        lines = f.readlines()
        indices_start_line = 0
        for line in lines[indices_start_line:]:
            idx = line.rfind(' ')
            if idx == -1:
                word = line.strip()
                count = 1
                self.indices[word] = len(self.symbols)
                self.symbols.append(word)
                self.count.append(count)
            else:
                word = line[:idx]
                count = int(line[idx + 1:])
                self.indices[word] = len(self.symbols)
                self.symbols.append(word)
                self.count.append(count)
    

    def index(self, sym):
        """Returns the index of the specified symbol"""
        assert isinstance(sym, str)
        if sym in self.indices:
            return self.indices[sym]
        return self.unk_index

    def encode_line(self, line, line_tokenizer=tokenize_line, append_eos=True, reverse_order=False):
        words = line_tokenizer(line)
        if reverse_order:
            words = list(reversed(words))
        nwords = len(words)
        ids = torch.IntTensor(nwords + 1 if append_eos else nwords)

        for i, word in enumerate(words):
            # if add_if_not_exist:
            #     idx = self.add_symbol(word)
            # else:
            idx = self.index(word)
            ids[i] = idx
        if append_eos:
            ids[nwords] = self.eos_index
        return ids


    def deocde_list(self, tensor_list):
        word_line = " ".join([self.__getitem__(w) for w in tensor_list])
        return word_line

    
    @staticmethod
    def _save_vocab_file(tokenized_sent_path, vocab_file):
        vocab = defaultdict(int)
        with open(tokenized_sent_path, "r") as f:
            for line in f:
                words = line.strip().lower().split()
                for word in words:
                    vocab[word] += 1

        sort_vocab = sorted(vocab.items(), key=lambda item: item[1], reverse=True)
        print("vocabulary size: {}, load from {}".format(len(vocab), tokenized_sent_path))
        # print(len(vocab), sort_vocab[-100:]) # bpe iters: 20000 vocab_size=14051, bpe iters: 10000 vocab_size=9252, no bpe vocab_size=16389
        with open(vocab_file, "w") as f:
            for (word, cnt) in sort_vocab:
                f.write(word + " " + str(cnt) + "\n")

if __name__ == "__main__":
    # tokenized_sent_path = "data/text2gloss/how2sign.train.pre.en"
    tokenized_sent_path = "how2sign.train.norm.tok.en"
    # tokenized_sent_path = "data/text2gloss/how2sign.train.norm.tok.clean.tc.en"
    vocab_file = "how2sign_vocab.txt"
    vocabulary = Dictionary()
    vocabulary._save_vocab_file(tokenized_sent_path, vocab_file)
    vocabulary = vocabulary.load(vocab_file)

    print(len(vocabulary))