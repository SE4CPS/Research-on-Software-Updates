from __future__ import annotations

from evaluation.config import EvaluationConfig
from evaluation.models.base import BaseDiscourseModel
from evaluation.models.logistic_regression import LogisticRegressionModel
from evaluation.models.majority import MajorityModel
from evaluation.models.naive_bayes import NaiveBayesModel
from evaluation.models.vader_rules import VaderRulesModel

try:
    from evaluation.models.roberta import RoBERTaModel

    _HAS_ROBERTA = True
except ImportError:
    _HAS_ROBERTA = False
    RoBERTaModel = None  # type: ignore

_REGISTRY: dict[str, type[BaseDiscourseModel]] = {
    "majority": MajorityModel,
    "logistic_regression": LogisticRegressionModel,
    "naive_bayes": NaiveBayesModel,
    "vader_rules": VaderRulesModel,
}
if _HAS_ROBERTA:
    _REGISTRY["roberta"] = RoBERTaModel


def list_models() -> list[str]:
    return sorted(_REGISTRY.keys())


def build_model(name: str, config: EvaluationConfig) -> BaseDiscourseModel:
    if name not in _REGISTRY:
        raise ValueError(f"Unknown model '{name}'. Choose from: {list_models()}")
    if name == "majority":
        return MajorityModel()
    if name == "logistic_regression":
        return LogisticRegressionModel(
            text_column=config.text_column,
            max_features=config.max_tfidf_features,
            random_state=config.random_state,
        )
    if name == "naive_bayes":
        return NaiveBayesModel(
            text_column=config.text_column,
            max_features=config.max_tfidf_features,
        )
    if name == "vader_rules":
        return VaderRulesModel(
            text_column=config.vader_text_column,
            neutral_band=config.vader_neutral_band,
            rule_style=config.vader_rule_style,
        )
    if name == "roberta":
        if not _HAS_ROBERTA:
            raise ImportError(
                "roberta requires: pip install torch transformers accelerate"
            )
        return RoBERTaModel(
            text_column=config.roberta_text_column,
            model_name=config.roberta_model_name,
            max_length=config.roberta_max_length,
            epochs=config.roberta_epochs,
            batch_size=config.roberta_batch_size,
            learning_rate=config.roberta_learning_rate,
            random_state=config.random_state,
        )
    raise ValueError(name)
