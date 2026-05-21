"""ML-based signal confidence scorer.

Loads a pre-trained RandomForest model and returns a WIN probability
for any FinalSignal. Integrates with the live engine as an optional layer.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

from .feature_extractor import extract_features, features_to_row
from .models import FinalSignal

logger = logging.getLogger(__name__)

_DEFAULT_MODEL_DIR = Path(__file__).parent.parent.parent / "data" / "models"


class MLPredictor:
    """Wraps a joblib-serialised sklearn classifier to score live signals."""

    def __init__(self, model_path: str | Path) -> None:
        try:
            import joblib  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError(
                "joblib is required. Install ML deps: pip install -e '.[ml]'"
            ) from exc

        self._model = joblib.load(model_path)
        meta_path = Path(model_path).with_suffix(".json")
        self._meta: dict = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        self._win_class_idx = self._resolve_win_idx()
        logger.info("MLPredictor loaded from %s", model_path)

    def _resolve_win_idx(self) -> int:
        classes = list(self._model.classes_)
        return classes.index("WIN") if "WIN" in classes else 1

    def predict_proba(self, signal: FinalSignal) -> float:
        """Return WIN probability in [0, 1] for the given signal."""
        features = extract_features(signal)
        row = [features_to_row(features)]
        proba = self._model.predict_proba(row)[0]
        return float(proba[self._win_class_idx])

    def predict_from_dict(self, features: dict[str, float]) -> float:
        """Return WIN probability from a pre-extracted feature dict."""
        row = [features_to_row(features)]
        proba = self._model.predict_proba(row)[0]
        return float(proba[self._win_class_idx])

    @property
    def meta(self) -> dict:
        return self._meta

    @classmethod
    def for_asset(
        cls,
        asset: str,
        timeframe: str = "1h",
        model_dir: str | Path | None = None,
    ) -> MLPredictor:
        """Convenience constructor: load the model for a given asset+timeframe."""
        base = Path(model_dir) if model_dir else _DEFAULT_MODEL_DIR
        safe = asset.replace("/", "_")
        path = base / f"{safe}_{timeframe}_rf.pkl"
        if not path.exists():
            raise FileNotFoundError(
                f"No trained model found at {path}. "
                f"Run: python scripts/train_model.py --asset {asset} --timeframe {timeframe}"
            )
        return cls(path)
