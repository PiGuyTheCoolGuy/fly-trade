"""Sparse expansion + learned ridge readout; optional real-connectome reservoir."""

from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import tempfile

import numpy as np
from scipy import sparse
from scipy.linalg import solve

from .config import Model
from .features import FEATURE_VERSION


class FlyModel:
    def __init__(self, settings: Model, input_size: int, graph_path: Path | None = None):
        self.settings = settings
        self.input_size = input_size
        self.graph_path = graph_path
        rng = np.random.default_rng(settings.seed)
        # Random sparse projections are an analogy to mushroom-body expansion;
        # they are not measured fly neurons or a whole-brain simulation.
        fan_in = min(6, input_size)
        self.projection = np.zeros((input_size, settings.hidden_size))
        for col in range(settings.hidden_size):
            selected = rng.choice(input_size, fan_in, replace=False)
            self.projection[selected, col] = rng.choice([-1.0, 1.0], fan_in) / np.sqrt(fan_in)
        self.mean = np.zeros(input_size)
        self.scale = np.ones(input_size)
        self.weights = None
        self.metadata = {"feature_version": FEATURE_VERSION}
        self.graph = None
        if settings.kind == "flywire":
            if graph_path is None or not graph_path.exists():
                raise ValueError("FlyWire graph missing; run: python run.py brain-download")
            self.graph = sparse.load_npz(graph_path).tocsr().astype(np.float64)
            if self.graph.shape[0] < 8 or self.graph.shape[0] != self.graph.shape[1]:
                raise ValueError("Invalid connectome graph")
            n = self.graph.shape[0]
            self.input_channels = rng.integers(input_size, size=n)
            self.input_sign = rng.choice([-1.0, 1.0], size=n)
            self.pool = np.arange(n) % settings.hidden_size
            self.pool_count = np.maximum(1, np.bincount(self.pool, minlength=settings.hidden_size))
            self.metadata["graph_sha256"] = hashlib.sha256(graph_path.read_bytes()).hexdigest()

    def encode(self, x: np.ndarray) -> np.ndarray:
        z = np.clip((x - self.mean) / self.scale, -8, 8)
        if self.graph is None:
            hidden = np.maximum(z @ self.projection, 0)
            k = max(1, int(self.settings.hidden_size * self.settings.active_fraction))
            cutoff = np.partition(hidden, -k, axis=1)[:, -k, None]
            hidden = np.where(hidden >= cutoff, hidden, 0)
        else:
            # Reset state per observation: a bounded graph transform, identical
            # offline/online. Deliberately not a biophysical or Shiu LIF replay.
            hidden = np.empty((len(z), self.settings.hidden_size))
            for i, row in enumerate(z):
                stimulus = row[self.input_channels] * self.input_sign * 0.25
                state = np.tanh(stimulus)
                for _ in range(self.settings.brain_steps):
                    state = np.tanh(stimulus + self.graph @ state)
                hidden[i] = np.bincount(self.pool, weights=state, minlength=self.settings.hidden_size) / self.pool_count
        return np.column_stack([np.ones(len(z)), z, hidden])

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        if len(x) < 100 or not np.isfinite(x).all() or not np.isfinite(y).all():
            raise ValueError("Need at least 100 finite training observations")
        self.mean = x.mean(axis=0)
        self.scale = np.maximum(x.std(axis=0), 1e-8)
        encoded = self.encode(x)
        gram = encoded.T @ encoded
        penalty = np.eye(gram.shape[0]) * self.settings.ridge
        penalty[0, 0] = 0
        self.weights = solve(gram + penalty, encoded.T @ y, assume_a="pos")

    def predict(self, x: np.ndarray) -> np.ndarray:
        if self.weights is None:
            raise ValueError("Model has not been trained")
        return np.clip(self.encode(x) @ self.weights, -0.25, 0.25)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        meta = {**self.metadata, "settings": asdict(self.settings), "input_size": self.input_size,
                "graph_path": str(self.graph_path.resolve()) if self.graph_path else None}
        fd, name = tempfile.mkstemp(dir=path.parent, suffix=".npz")
        try:
            with os.fdopen(fd, "wb") as out:
                np.savez_compressed(out, mean=self.mean, scale=self.scale, weights=self.weights,
                                    projection=self.projection, metadata=json.dumps(meta))
            os.replace(name, path)
        finally:
            if os.path.exists(name):
                os.unlink(name)

    @classmethod
    def load(cls, path: Path, graph_path: Path | None = None):
        # No pickle/joblib: models contain arrays and JSON, never executable objects.
        with np.load(path, allow_pickle=False) as data:
            meta = json.loads(str(data["metadata"]))
            if meta["feature_version"] != FEATURE_VERSION:
                raise ValueError("Model feature version changed; retrain")
            graph = graph_path or (Path(meta["graph_path"]) if meta["graph_path"] else None)
            model = cls(Model(**meta["settings"]), meta["input_size"], graph)
            if model.graph is not None and model.metadata["graph_sha256"] != meta["graph_sha256"]:
                raise ValueError("Connectome changed; retrain before paper trading")
            model.metadata = meta
            for key in ("mean", "scale", "weights", "projection"):
                setattr(model, key, data[key].copy())
        return model
