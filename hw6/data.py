from collections import Counter
from typing import Any, Optional

import numpy as np
from vocabulary import Vocabulary

# type aliases
Example = dict[str, Any]


class Dataset:
    def __init__(self, examples: list[Example], vocab: Vocabulary) -> None:
        self.examples = examples
        self.vocab = vocab

    def example_to_tensors(self, index: int) -> dict[str, np.ndarray]:
        raise NotImplementedError

    def batch_as_tensors(self, start: int, end: int) -> dict[str, np.ndarray]:
        raise NotImplementedError

    def __getitem__(self, idx: int) -> dict:
        return self.examples[idx]

    def __len__(self) -> int:
        return len(self.examples)


def pad_batch(sequences: list[np.ndarray], padding_index: int) -> np.ndarray:
    """Pad a list of sequences, so that that they all have the same length.
    Return as one [batch_size, max_seq_len] numpy array.

    Example usage:
    >>> pad_batch([np.array([2, 4]), np.array([1, 3, 6]), np.array([2])], 0)
    >>> np.array([[2, 4, 0], [1, 3, 6], [2, 0, 0]])

    Arguments:
        sequences: list of arrays, each containing integer indices, to pad and combine
            Each array will be 1-dimensional, but they might have different lengths.
        padding_index: integer index of PAD symbol, used to fill in to make sequences longer

    Returns:
        [batch_size, max_seq_len] numpy array, where each row is the corresponding
        sequence from the sequences argument, but padded out to the maximum length with
        padding_index
    """
    # TODO: implement here! ~6-7 lines
    # NB: you should _not_ directly modify the `sequences` argument, or any of the arrays
    # contained in that list.  (This may cause tests, and some aspects of training, to fail.)
    # Rather, you can use them without modification to build your new array to be returned.
    max_len = max(len(seq) for seq in sequences)
    batch_size = len(sequences)
    padded = np.full((batch_size, max_len), padding_index, dtype=np.int32)
    for i, seq in enumerate(sequences):
        padded[i, :len(seq)] = seq
    return padded


class SSTClassificationDataset(Dataset):

    labels_to_string = {0: "terrible", 1: "bad", 2: "so-so", 3: "good", 4: "excellent"}
    label_one_hots = np.eye(len(labels_to_string))
    PAD = "<PAD>"

    def example_to_tensors(self, index: int) -> dict[str, np.ndarray]:
        example = self.__getitem__(index)
        return {
            "review": np.array(self.vocab.tokens_to_indices(example["review"])),
            "label": example["label"],
        }

    def batch_as_tensors(self, start: int, end: int) -> dict[str, np.ndarray]:
        examples = [self.example_to_tensors(index) for index in range(start, end)]
        padding_index = self.vocab[SSTClassificationDataset.PAD]
        return {
            "review": pad_batch(
                [example["review"] for example in examples], padding_index
            ),
            "label": np.array([example["label"] for example in examples]),
            "lengths": np.array([len(example["review"]) for example in examples]),
        }

    @classmethod
    def from_files(cls, reviews_file: str, labels_file: str, vocab: Vocabulary = None):
        with open(reviews_file, "r") as reviews, open(labels_file, "r") as labels:
            review_lines = reviews.readlines()
            label_lines = labels.readlines()
        examples = [
            {
                # review is text, stored as a list of tokens
                "review": review_lines[line].strip("\n").split(" "),
                "label": int(label_lines[line].strip("\n")),
            }
            for line in range(len(review_lines))
        ]
        # initialize a vocabulary from the reviews, if none is given
        if not vocab:
            vocab = Vocabulary.from_text_files(
                [reviews_file],
                special_tokens=(Vocabulary.UNK, SSTClassificationDataset.PAD),
            )
        return cls(examples, vocab)


#####
# PART 2, Decoder / LM data
#####


def example_from_characters(characters: list[str], bos: str, eos: str) -> Example:
    """Generate a sequence of language modeling targets from a list of characters.

    Example usage:
    >>> example_from_characters(['a', 'b', 'c'], '<s>', '</s>')
    >>> {'text': ['<s>', 'a', 'b', 'c'], 'target': ['a', 'b', 'c', '</s>'], 'length': 4}

    Arguments:
        characters: a list of strings, the characters in a sequence
        bos: beginning of sequence symbol, to be prepended as an input
        eos: end of sequence symbol, to be appended as a target

    Returns:
        an Example dictionary, as given in the example above, with three fields:
        text, target, and length
    """
    # TODO: implement here (~2-3 lines)
    text = [bos] + characters
    target = characters + [eos]
    return {"text": text, "target": target, "length": len(text)}


class SSTLanguageModelingDataset(Dataset):

    BOS = "<s>"
    EOS = "</s>"
    PAD = "<PAD>"

    def __init__(self, examples: list[Example], vocab: Vocabulary) -> None:
        super().__init__(examples, vocab)
        self.num_labels = len(self.vocab)
        self._label_one_hots = np.eye(self.num_labels)

    def example_to_indices(self, index: int) -> dict[str, Any]:
        example = self.__getitem__(index)
        return {
            "text": np.array(self.vocab.tokens_to_indices(example["text"])),
            "target": self.vocab.tokens_to_indices(example["target"]),
            "length": example["length"],
        }

    def batch_as_tensors(self, start: int, end: int) -> dict[str, Any]:
        examples = [self.example_to_indices(index) for index in range(start, end)]
        padding_index = self.vocab[SSTLanguageModelingDataset.PAD]
        # pad texts to [batch_size, max_seq_len] array
        texts = pad_batch(
            [np.array(example["text"]) for example in examples], padding_index
        )
        # target: [batch_size, max_seq_len], indices for next character
        target = pad_batch(
            [np.array(example["target"]) for example in examples], padding_index
        )
        return {
            "text": texts,
            "target": target,
            "length": [example["length"] for example in examples],
        }

    @classmethod
    def from_file(cls, text_file: str, vocab: Optional[Vocabulary] = None):
        examples = []
        counter: Counter = Counter()
        with open(text_file, "r") as reviews:
            for line in reviews:
                string = line.strip("\n")
                counter.update(string)
                # generate example from line
                examples.append(
                    example_from_characters(
                        list(string),
                        SSTLanguageModelingDataset.BOS,
                        SSTLanguageModelingDataset.EOS,
                    )
                )
        if not vocab:
            vocab = Vocabulary(
                counter,
                special_tokens=(
                    Vocabulary.UNK,
                    SSTLanguageModelingDataset.BOS,
                    SSTLanguageModelingDataset.EOS,
                    SSTLanguageModelingDataset.PAD,
                ),
            )
        return cls(examples, vocab)
