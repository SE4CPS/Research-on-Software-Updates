"""
RoBERTa fine-tuning for TPS/GDS within cross-validation folds.
Uses a minimal PyTorch training loop (no HuggingFace Trainer / accelerate).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from evaluation.models.base import BaseDiscourseModel


def _pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class _TextLabelDataset(Dataset):
    def __init__(self, texts: list[str], labels: np.ndarray, tokenizer, max_length: int):
        self.texts = texts
        self.labels = labels.astype(int)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        enc = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        item = {k: v.squeeze(0) for k, v in enc.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


class RoBERTaModel(BaseDiscourseModel):
    name = "roberta"

    def __init__(
        self,
        text_column: str = "text_raw",
        model_name: str = "roberta-base",
        max_length: int = 256,
        epochs: int = 2,
        batch_size: int = 8,
        learning_rate: float = 2e-5,
        weight_decay: float = 0.01,
        random_state: int = 42,
    ) -> None:
        self.text_column = text_column
        self.model_name = model_name
        self.max_length = max_length
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.random_state = random_state
        self._model = None
        self._tokenizer = None
        self._device = _pick_device()

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> None:
        texts = X[self.text_column].fillna("").astype(str).tolist()
        if not any(t.strip() for t in texts):
            raise ValueError(f"Empty text in column {self.text_column}")

        torch.manual_seed(self.random_state)
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=2,
        )
        self._model.to(self._device)
        self._model.train()

        y_int = y.astype(int)
        classes = np.unique(y_int)
        weights = compute_class_weight("balanced", classes=classes, y=y_int)
        class_weights = torch.tensor(weights, dtype=torch.float32, device=self._device)

        train_ds = _TextLabelDataset(texts, y_int, self._tokenizer, self.max_length)
        loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True)

        optimizer = torch.optim.AdamW(
            self._model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        for _epoch in range(self.epochs):
            for batch in loader:
                input_ids = batch["input_ids"].to(self._device)
                attention_mask = batch["attention_mask"].to(self._device)
                labels = batch["labels"].to(self._device)
                optimizer.zero_grad()
                outputs = self._model(input_ids=input_ids, attention_mask=attention_mask)
                loss = torch.nn.functional.cross_entropy(
                    outputs.logits, labels, weight=class_weights
                )
                loss.backward()
                optimizer.step()

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        proba = self.predict_proba(X)
        return np.argmax(proba, axis=1).astype(int)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if self._model is None or self._tokenizer is None:
            raise RuntimeError("Model not fitted")
        self._model.eval()
        texts = X[self.text_column].fillna("").astype(str).tolist()
        all_probs: list[np.ndarray] = []
        bs = self.batch_size
        with torch.no_grad():
            for i in range(0, len(texts), bs):
                batch = texts[i : i + bs]
                enc = self._tokenizer(
                    batch,
                    truncation=True,
                    padding=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                )
                enc = {k: v.to(self._device) for k, v in enc.items()}
                logits = self._model(**enc).logits
                probs = torch.softmax(logits, dim=-1).cpu().numpy()
                all_probs.append(probs)
        return np.vstack(all_probs)

    def get_params(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "text_column": self.text_column,
            "model_name": self.model_name,
            "max_length": self.max_length,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "device": str(self._device),
        }
