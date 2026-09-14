"""Download pinned upstream data and build an explicitly simplified reservoir."""

import hashlib
import logging
from pathlib import Path
import time
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd
from scipy import sparse

from .config import Config
from .storage import atomic_json

LOG = logging.getLogger(__name__)
UPSTREAM = "https://github.com/philshiu/Drosophila_brain_model"
REVISION = "91bdd1e7dcf193f3e7ca5a8933497fcef63b7960"
FILES = {
    "Completeness_783.csv": (3327347, "b5a26b82b69a3c2e7fd99cd6810c36fc8f0e492b"),
    "Connectivity_783.parquet": (100804642, "d386555d1a5f40ebfa1380bcb05b1fab044855fd"),
}


def graph_path(config: Config) -> Path:
    return config.data_dir / "brain" / f"flywire-{config.model.brain_neurons}.npz"


def _valid(path: Path, size: int, blob_sha: str) -> bool:
    if not path.exists() or path.stat().st_size != size:
        return False
    digest = hashlib.sha1(f"blob {size}\0".encode())
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest() == blob_sha


def download_file(directory: Path, name: str) -> Path:
    size, digest = FILES[name]
    path = directory / name
    if _valid(path, size, digest):
        return path
    url = f"https://raw.githubusercontent.com/philshiu/Drosophila_brain_model/{REVISION}/{name}"
    partial = path.with_suffix(path.suffix + ".partial")
    for attempt in range(3):
        try:
            LOG.info("Downloading %s (%.1f MB)", name, size / 1e6)
            req = Request(url, headers={"User-Agent": "FlyTrade/0.1"})
            with urlopen(req, timeout=45) as response, partial.open("wb") as out:
                count = 0
                while block := response.read(1024 * 1024):
                    count += len(block)
                    if count > size:
                        raise ValueError("Upstream data size changed")
                    out.write(block)
            if not _valid(partial, size, digest):
                raise ValueError("Upstream data checksum mismatch")
            partial.replace(path)
            return path
        except (OSError, ValueError):
            partial.unlink(missing_ok=True)
            if attempt == 2:
                raise
            time.sleep(2 ** attempt)
    raise RuntimeError("Download failed")


def prepare_graph(config: Config) -> Path:
    try:
        import pyarrow  # noqa: F401
    except ImportError as exc:
        raise ValueError('Install connectome support: python -m pip install -e ".[connectome]"') from exc
    directory = config.data_dir / "brain"
    directory.mkdir(parents=True, exist_ok=True)
    nodes = download_file(directory, "Completeness_783.csv")
    edges = download_file(directory, "Connectivity_783.parquet")
    ids = pd.read_csv(nodes, index_col=0).index.to_numpy()
    connections = pd.read_parquet(edges, columns=["Presynaptic_Index", "Postsynaptic_Index", "Excitatory x Connectivity"])
    pre = connections["Presynaptic_Index"].to_numpy(dtype=np.int64)
    post = connections["Postsynaptic_Index"].to_numpy(dtype=np.int64)
    weights = connections["Excitatory x Connectivity"].to_numpy(dtype=float)
    n = len(ids)
    if (pre < 0).any() or (post < 0).any() or max(pre.max(), post.max()) >= n or not np.isfinite(weights).all():
        raise ValueError("Connectome schema/index mismatch")
    count = config.model.brain_neurons or n
    if count < 8:
        raise ValueError("brain_neurons must be at least 8, or 0 for the full graph")
    degree = np.bincount(pre, weights=np.abs(weights), minlength=n) + np.bincount(post, weights=np.abs(weights), minlength=n)
    selected = np.sort(np.lexsort((np.arange(n), -degree))[:min(count, n)])
    remap = np.full(n, -1, dtype=np.int64)
    remap[selected] = np.arange(len(selected))
    mask = (remap[pre] >= 0) & (remap[post] >= 0)
    # A[post, pre] preserves direction; duplicate connections are summed.
    graph = sparse.coo_matrix((weights[mask], (remap[post[mask]], remap[pre[mask]])),
                              shape=(len(selected), len(selected))).tocsr()
    graph.eliminate_zeros()
    if graph.nnz == 0:
        raise ValueError("Selected graph contains no edges")
    # Row L1 <= .8 makes the recurrent transform contractive and numerically bounded.
    norm = np.asarray(abs(graph).sum(axis=1)).ravel()
    graph = (sparse.diags(0.8 / np.maximum(norm, 1)) @ graph).tocsr()
    output = graph_path(config)
    temp = output.with_suffix(".tmp.npz")
    sparse.save_npz(temp, graph)
    temp.replace(output)
    atomic_json(output.with_suffix(".json"), {
        "upstream": UPSTREAM, "revision": REVISION, "dataset": "FlyWire v783",
        "source_neurons": n, "selected_neurons": len(selected), "edges": graph.nnz,
        "selection": "all" if len(selected) == n else "induced subgraph of highest weighted-degree neurons",
        "transform": "signed directed graph; row L1 normalized to <=0.8; tanh dynamics",
        "not_a_biophysical_simulation": True,
        "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
    })
    np.save(output.with_suffix(".node_ids.npy"), ids[selected], allow_pickle=False)
    LOG.info("FlyWire graph ready: %d neurons, %d edges", len(selected), graph.nnz)
    return output
