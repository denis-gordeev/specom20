import numpy as np
import torch
from dataclasses import dataclass
from typing import List, Optional, Union
from transformers import (
    BertConfig,
    BertModel,
    BertForSequenceClassification,
    BertTokenizer,
)
from torch.utils.data import (
    DataLoader,
    RandomSampler,
    SequentialSampler,
    TensorDataset,
)
import logging
from ranepa_flask_wrapper.flask_wrapper import flask_wrapper

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


def load_and_cache_examples(texts: List[str], tokenizer):
    output_mode = "classification"
    label_list = [0, 1]
    examples = [InputExample(guid=t_i, text=t) for t_i, t in enumerate(texts)]
    features = convert_examples_to_features(
        examples,
        tokenizer,
        label_list=label_list,
        max_seq_length=128,
        output_mode=output_mode,
        pad_on_left=False,
        pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
        pad_token_segment_id=4,
    )
    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor(
        [f.input_ids for f in features], dtype=torch.long
    )
    all_attention_mask = torch.tensor(
        [f.input_mask for f in features], dtype=torch.long
    )
    all_token_type_ids = torch.tensor(
        [f.segment_ids for f in features], dtype=torch.long
    )
    if output_mode == "classification":
        all_labels = torch.tensor(
            [f.label for f in features], dtype=torch.long
        )
    elif output_mode == "regression":
        all_labels = torch.tensor(
            [f.label for f in features], dtype=torch.float
        )
    dataset = TensorDataset(
        all_input_ids, all_attention_mask, all_token_type_ids, all_labels,
    )
    return dataset


class BertPredictor:
    def __init__(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        label_list = [0, 1]
        config_class, model_class, tokenizer_class = (
            BertConfig,
            BertForSequenceClassification,
            BertTokenizer,
        )
        config = config_class.from_pretrained("./trained-model/config.json")
        config.num_labels = 2
        tokenizer = tokenizer_class.from_pretrained("./trained-model/")
        model = BertForSequenceClassification.from_pretrained(
            "./trained-model/"
        )
        model.to(device)
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    def predict(self, texts):
        output_mode = "classification"
        eval_dataset = load_and_cache_examples(texts, self.tokenizer)
        eval_batch_size = 96
        eval_sampler = SequentialSampler(eval_dataset)
        eval_dataloader = DataLoader(
            eval_dataset, sampler=eval_sampler, batch_size=eval_batch_size
        )

        eval_loss = 0.0
        nb_eval_steps = 0
        preds = None
        out_label_ids = None
        for batch in eval_dataloader:
            self.model.eval()
            batch = tuple(t.to(self.device) for t in batch)
            with torch.no_grad():
                inputs = {
                    "input_ids": batch[0],
                    "attention_mask": batch[1],
                    "labels": batch[3],
                }
                outputs = self.model(**inputs)
                tmp_eval_loss, logits = outputs[:3]
            nb_eval_steps += 1
            if preds is None:
                preds = logits.detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
        if output_mode == "classification":
            preds = np.argmax(preds, axis=1)
        elif output_mode == "regression":
            preds = np.squeeze(preds)
        list_preds = [int(p) for p in preds]
        return list_preds


if __name__ == "__main__":
    bert_predictor = BertPredictor()
    flask_wrapper(bert_predictor.predict, port=5098, host="0.0.0.0")
