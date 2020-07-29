# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and
# The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Classification fine-tuning: utilities to work with various datasets """

from __future__ import absolute_import, division, print_function

import logging
import os
import re
from dataclasses import dataclass
from glob import glob
from typing import List, Optional, Union

import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


@dataclass
class InputExample(object):
    """Constructs a InputExample.

    Args:
        guid: Unique id for the example.
        text: string. The untokenized text of the sequence
        label: (Optional) string. The label of the example. This should be
        specified for train and dev examples, but not for test examples.
    """

    guid: Union[str, int]
    text: str
    label: Optional[str] = None


@dataclass
class InputFeatures(object):
    """A single set of features of data."""

    input_ids: List[int]
    input_mask: List[int]
    segment_ids: List[int]
    label: int


class PanicProcessor(object):
    """Processor for the Panic data set ."""

    def __init__(self, folder_list: Optional[List[str]] = None):
        """Init PanicProcessor
        
        Keyword Arguments:
            folder_list {Optional[List[str]]} -- [description] (default: {None})
        """
        if folder_list:
            self.folder_list = folder_list
        else:
            self.folder_list = ["eng", "hin", "iben"]

    def get_labels(self):
        return [0, 1]

    """def get_label_from_name(self, name):
        return label_dict[id]"""

    def clean_text(self, text):
        text = re.sub("post:.*", "", text).strip()
        text = re.sub("parent:.*", "", text).strip()
        text = [w.strip() for w in text.split("\n")]
        text = "".join([w for w in text][:1])
        text = text[:3000]
        return text

    def get_ranepa_examples(self, data_dir, set_type):
        df = pd.read_csv("doccano3.csv")
        individual_files = glob("1*.csv")
        for filename in individual_files:
            new_df = pd.read_csv(filename)
            print(filename, new_df.shape[0])
            df = df.append(new_df)

        df["text"] = df["text"].apply(lambda x: self.clean_text(x))
        # df = df[df.groupby(["text", "label"]).transform("count")["user"] >= 2]
        df = df.groupby("text").first().reset_index()

        df["label"] = df["label"].astype(int)
        df = df[df["label"].isin({282, 283})]
        df["label"] = df["label"].replace(283, 0)
        df["label"] = df["label"].replace(282, 1)

    def prepare_tolokka_all_examples(self):
        df = pd.read_csv(f"toloka_panic_nonblocked.tsv", sep="\t")
        df = df[
            df.groupby(["INPUT:Комментарий", "OUTPUT:result"]).transform(
                "count"
            )["INPUT:reply_to_post"]
            >= 3
        ]
        df = df.groupby(["INPUT:Комментарий"]).first().reset_index()
        df["OUTPUT:result"] = df["OUTPUT:result"].replace(
            {"PANIC": 1, "NOT_PANIC": 0}
        )
        train, dev = train_test_split(df)
        train = train.append(
            [train[train["OUTPUT:result"] == 1]] * 5, ignore_index=True
        )
        train.to_csv(f"toloka_panic_nonblocked_train.tsv", index=False)
        dev.to_csv(f"toloka_panic_nonblocked_dev.tsv", index=False)
        return examples

    # def get_tolokka_all_examples(self, data_dir, set_type):
    def get_examples(self, data_dir, set_type):
        df = pd.read_csv(f"kaggle_shortened_binary_{set_type}.csv")
        examples = []
        for row_id, row in df.iterrows():
            example = InputExample(
                guid=row_id,
                text=row["comment_text"],
                label=row.get("severe_toxicity"),
            )
            examples.append(example)
        return examples

    def get_tolokka_agg_examples(self, data_dir, set_type):
        """Creates examples for the training and dev sets from labelled folder."""
        examples = []
        df = pd.read_csv(f"toloka_panic_aggregated_{set_type}.tsv", sep="\t")
        df = df[(df["CONFIDENCE:result"] > 80)]
        df = df.groupby(["INPUT:Комментарий"]).first().reset_index()

        if set_type == "train":
            df = df.append(
                [df[df["OUTPUT:result"] == 1]] * 5, ignore_index=True
            )
        for row_id, row in df.iterrows():
            example = InputExample(
                guid=row_id,
                text=row["INPUT:Комментарий"],
                label=row.get("OUTPUT:result"),
            )
            examples.append(example)
        return examples


def convert_examples_to_features(
    examples,
    tokenizer,
    label_list,
    max_seq_length,
    output_mode,
    cls_token_at_end=False,
    pad_on_left=False,
    cls_token="[CLS]",
    sep_token="[SEP]",
    pad_token=0,
    sequence_segment_id=0,
    cls_token_segment_id=1,
    pad_token_segment_id=0,
    mask_padding_with_zero=True,
):
    """ Loads a data file into a list of `InputBatch`s
        `cls_token_at_end` define the location of the CLS token:
            - False (Default, BERT/XLM pattern): [CLS] + A + [SEP] + B + [SEP]
            - True (XLNet/GPT pattern): A + [SEP] + B + [SEP] + [CLS]
        `cls_token_segment_id` define the segment id associated to the CLS token (0 for BERT, 2 for XLNet)
    """
    label_maps = {label: i for i, label in enumerate(label_list)}

    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))

        tokens = tokenizer.tokenize(example.text)
        tokens = tokens[: (max_seq_length - 2)]

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids:   0   0  0    0    0     0       0   0   1  1  1  1   1   1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids:   0   0   0   0  0     0   0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambiguously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens = tokens + [sep_token]
        segment_ids = [sequence_segment_id] * len(tokens)

        if cls_token_at_end:
            tokens = tokens + [cls_token]
            segment_ids = segment_ids + [cls_token_segment_id]
        else:
            tokens = [cls_token] + tokens
            segment_ids = [cls_token_segment_id] + segment_ids

        # padding_length = max_seq_length - len(input_ids)
        sequence_a_dict = tokenizer.encode_plus(
            tokens, max_length=max_seq_length, pad_to_max_length=True
        )
        input_ids = sequence_a_dict["input_ids"]

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        attention_mask = sequence_a_dict["attention_mask"]

        # Zero-pad up to the sequence length.
        padding_length = max_seq_length - len(segment_ids)
        if pad_on_left:
            segment_ids = (
                [pad_token_segment_id] * padding_length
            ) + segment_ids
        else:
            segment_ids = segment_ids + (
                [pad_token_segment_id] * padding_length
            )

        assert len(input_ids) == max_seq_length
        assert len(attention_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        if output_mode == "classification":
            label = label_maps.get(example.label)
            if label is None:
                label = 0
        elif output_mode == "regression":
            label_id = float(example.label)
        else:
            raise KeyError(output_mode)

        if ex_index < 0:
            logger.info("*** Example ***")
            logger.info("guid: {}".format(example.guid))
            logger.info(
                "tokens: {}".format(" ".join([str(x) for x in tokens]))
            )
            logger.info(
                "input_ids: {}".format(" ".join([str(x) for x in input_ids]))
            )
            logger.info(
                "input_mask: {}".format(
                    " ".join([str(x) for x in attention_mask])
                )
            )
            logger.info(
                "segment_ids: {}".format(
                    " ".join([str(x) for x in segment_ids])
                )
            )
            logger.info("label: {} (id = {})".format(example.label, label_id))

        features.append(
            InputFeatures(
                input_ids=input_ids,
                input_mask=attention_mask,
                segment_ids=segment_ids,
                label=label,
            )
        )
    return features


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def simple_accuracy(preds, labels):
    return (preds == labels).mean()


def acc_and_f1(preds, labels, average="micro"):
    acc = simple_accuracy(preds, labels)
    f1 = f1_score(y_true=labels, y_pred=preds, average=average)
    f1_micro = f1_score(y_true=labels, y_pred=preds, average="micro")
    f1_macro = f1_score(y_true=labels, y_pred=preds, average="macro")
    precision = precision_score(y_true=labels, y_pred=preds)
    recall = recall_score(y_true=labels, y_pred=preds)
    return {
        "acc": acc,
        "f1": f1,
        "f1_micro": f1_micro,
        "f1_macro": f1_macro,
        "acc_and_f1": (acc + f1) / 2,
        "recall": recall,
        "precision": precision,
    }


def pearson_and_spearman(preds, labels):
    pearson_corr = pearsonr(preds, labels)[0]
    spearman_corr = spearmanr(preds, labels)[0]
    return {
        "pearson": pearson_corr,
        "spearmanr": spearman_corr,
        "corr": (pearson_corr + spearman_corr) / 2,
    }


def compute_metrics(task_name, preds, labels):
    assert len(preds) == len(labels)
    if task_name in {"imdb", "quora", "trac", "panic"}:
        return acc_and_f1(preds, labels)
    else:
        raise KeyError(task_name)


processors = {"panic": PanicProcessor}

output_modes = {"panic": "classification"}
