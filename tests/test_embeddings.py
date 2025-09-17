import numpy as np
import torch
from pathlib import Path
from types import SimpleNamespace
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.embeddings import get_hf_transformer_embeddings


class DummyBatch(dict):
    def to(self, device):
        self["input_ids"] = self["input_ids"].to(device)
        self["attention_mask"] = self["attention_mask"].to(device)
        return self


class DummyTokenizer:
    def __init__(self):
        self.pad_token = None
        self.pad_token_id = None
        self.eos_token = "<eos>"
        self.eos_token_id = 99
        self.sep_token = None
        self.cls_token = None
        self.bos_token = None
        self.unk_token = "<unk>"
        self.vocab = {
            "hello": [1, 2],
            "world": [3],
        }

    def convert_tokens_to_ids(self, token):
        if token == self.eos_token:
            return self.eos_token_id
        if token == self.unk_token:
            return 0
        raise KeyError(token)

    def __call__(self, batch, return_tensors, padding, truncation, max_length):
        assert return_tensors == "pt"
        assert padding is True
        assert truncation is True

        max_len = min(max(len(self.vocab[text]) for text in batch), max_length)
        input_ids = []
        attention_mask = []
        for text in batch:
            tokens = list(self.vocab[text])[:max_len]
            mask = [1] * len(tokens)
            pad_len = max_len - len(tokens)
            if pad_len > 0:
                pad_id = (
                    self.pad_token_id
                    if self.pad_token_id is not None
                    else self.convert_tokens_to_ids(self.eos_token)
                )
                tokens.extend([pad_id] * pad_len)
                mask.extend([0] * pad_len)
            input_ids.append(tokens)
            attention_mask.append(mask)

        return DummyBatch(
            {
                "input_ids": torch.tensor(input_ids, dtype=torch.long),
                "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            }
        )


class DummyModel:
    def to(self, device):
        self.device = device
        return self

    def __call__(self, input_ids=None, attention_mask=None):
        last_hidden_state = input_ids.unsqueeze(-1).float()
        return SimpleNamespace(last_hidden_state=last_hidden_state)


def test_get_hf_transformer_embeddings_without_pad_token(monkeypatch):
    dummy_tokenizer = DummyTokenizer()
    dummy_model = DummyModel()

    def fake_tokenizer_from_pretrained(model_name):
        return dummy_tokenizer

    def fake_model_from_pretrained(model_name):
        return dummy_model

    monkeypatch.setattr(
        "transformers.AutoTokenizer.from_pretrained", fake_tokenizer_from_pretrained
    )
    monkeypatch.setattr(
        "transformers.AutoModel.from_pretrained", fake_model_from_pretrained
    )

    texts = ["hello", "world"]
    embeddings = get_hf_transformer_embeddings(texts, "dummy-model", device="cpu")

    assert isinstance(embeddings, np.ndarray)
    assert embeddings.shape == (2, 1)
    np.testing.assert_allclose(embeddings, np.array([[1.5], [3.0]]))
    assert dummy_tokenizer.pad_token == dummy_tokenizer.eos_token
    assert dummy_tokenizer.pad_token_id == dummy_tokenizer.eos_token_id
