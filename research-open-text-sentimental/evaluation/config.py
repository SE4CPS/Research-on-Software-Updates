"""
Paths and constants for ICMLA evaluation infrastructure.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

# Repository root (research-open-text-sentimental/)
REPO_ROOT = Path(__file__).resolve().parent.parent
TPS_GDS_ROOT = REPO_ROOT / "tps_gds_classification"

DEFAULT_LABELS_CSV = TPS_GDS_ROOT / "data" / "updated_labeled_dataset_unique.csv"
DEFAULT_DATA_JSON = TPS_GDS_ROOT / "data" / "tps_gds_dataset.json"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "evaluation" / "outputs"
DEFAULT_C1_PRIME_CSV = REPO_ROOT / "data" / "c1_prime" / "c1_prime_metrics.csv"

# Paper Tier-A subreddit allowlist (lowercase keys for matching)
TIER_A_SUBREDDITS: frozenset[str] = frozenset(
    {
        "android",
        "chrome",
        "comfyui",
        "django",
        "docker",
        "firefox",
        "github",
        "godot",
        "ibm",
        "immich",
        "java",
        "kubernetes",
        "linux",
        "macos",
        "neovim",
        "nest",
        "node",
        "nuxt",
        "oracle",
        "php",
        "python",
        "react",
        "rust",
        "rustdesk",
        "supabase",
        "typescript",
        "vscode",
        "wordpress",
    }
)


@dataclass
class EvaluationConfig:
    """Runtime configuration for run_evaluation.py."""

    labels_path: Path = DEFAULT_LABELS_CSV
    data_json_path: Path = DEFAULT_DATA_JSON
    output_dir: Path = DEFAULT_OUTPUT_DIR
    random_state: int = 42
    n_folds: int = 5
    models: tuple[str, ...] = (
        "majority",
        "logistic_regression",
        "naive_bayes",
        "vader_rules",
    )
    text_column: str = "text"
    vader_text_column: str = "text_raw"
    undersample_gds: bool = False
    gds_sample_size: int = 175
    max_tfidf_features: int = 12_000
    bootstrap_samples: int = 1000
    bootstrap_ci: float = 0.95
    tier_a_only: bool = False
    vader_rule_style: str = "technical_first"
    vader_neutral_band: float = 0.15
    c1_prime_csv: Path = DEFAULT_C1_PRIME_CSV
    # RoBERTa (optional; slow — use --models roberta or paper_full run)
    roberta_model_name: str = "roberta-base"
    roberta_text_column: str = "text_raw"
    roberta_max_length: int = 256
    roberta_epochs: int = 2
    roberta_batch_size: int = 8
    roberta_learning_rate: float = 2e-5

    def resolved_output_dir(self, run_name: str | None = None) -> Path:
        if run_name:
            return self.output_dir / run_name
        return self.output_dir / "latest"
