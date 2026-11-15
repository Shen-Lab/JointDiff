"""Boltz-1 style structure/sequence data loader (no MSA, no profile features).

Mirrors ``boltz.data.module.training.TrainingDataset`` from Boltz v0.4.1 but:
  * never loads MSAs, and emits only token (sequence/structure) and atom features;
  * makes the sampler and the cropper optional.

Reuses Boltz's own tokenizer, samplers, cropper and token/atom featurizers
(``pip install boltz==0.4.1``) so the outputs match Boltz-1 exactly.

Modes
-----
sampler=None            -> deterministic pass over the records (len = #records)
sampler="random"|"cluster"|Sampler
                        -> infinite weighted sampling (len = samples_per_epoch)
cropper=None            -> full structures (use batch_size=1 or pad via collate)
cropper="boltz"|Cropper -> crop to max_tokens / max_atoms
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Callable, Iterable, Iterator, Optional, Union

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, get_worker_info

from boltz.data.crop.boltz import BoltzCropper
from boltz.data.crop.cropper import Cropper
from boltz.data.feature.featurizer import process_atom_features, process_token_features
from boltz.data.feature.pad import pad_to_max
from boltz.data.sample.cluster import ClusterSampler
from boltz.data.sample.random import RandomSampler
from boltz.data.sample.sampler import Sample, Sampler
from boltz.data.tokenize.boltz import BoltzTokenizer
from boltz.data.types import Connection, Input, Manifest, Record, Structure

# Features that are not stacked/padded by ``collate`` (variable-length per sample).
_NO_PAD_KEYS = {
    "all_coords",
    "all_resolved_mask",
    "crop_to_all_atom_map",
    "chain_symmetries",
    "amino_acids_symmetries",
    "ligand_symmetries",
}


def load_structure(record: Record, target_dir: Path) -> Structure:
    """Load a processed structure ``.npz`` (no MSA)."""
    s = np.load(target_dir / "structures" / f"{record.id}.npz")
    return Structure(
        atoms=s["atoms"],
        bonds=s["bonds"],
        residues=s["residues"],
        chains=s["chains"],
        connections=s["connections"].astype(Connection),
        interfaces=s["interfaces"],
        mask=s["mask"],
    )


def _read_ids(ids: Union[str, Path, Iterable[str], None]) -> Optional[set[str]]:
    if ids is None:
        return None
    if isinstance(ids, (str, Path)):
        ids = Path(ids).read_text().splitlines()
    return {x.strip().lower() for x in ids if x.strip()}


class _SequentialSampler(Sampler):
    """Deterministic pass over records, one epoch, no replacement."""

    def sample(self, records, random) -> Iterator[Sample]:  # noqa: ARG002
        for record in records:
            yield Sample(record=_valid_only(record))


def _valid_only(record: Record) -> Record:
    """Drop invalid chains/interfaces (as Boltz's samplers do)."""
    return replace(
        record,
        chains=[c for c in record.chains if c.valid],
        interfaces=[i for i in record.interfaces if i.valid],
    )


class StructureSequenceDataset(Dataset):
    """Boltz-1 style dataset yielding token + atom features only.

    Parameters
    ----------
    target_dir : path to ``rcsb_processed_targets`` (contains manifest.json, structures/).
    sampler : None | "random" | "cluster" | Sampler
        None iterates records in order (``len`` = number of records, ``idx`` picks the
        record). Otherwise samples infinitely and ``len`` = ``samples_per_epoch``;
        ``idx`` is ignored. "cluster" is the AF3-style chain/interface weighted
        sampler used by Boltz-1 on RCSB.
    cropper : None | "boltz" | Cropper
        None keeps whole structures. "boltz" uses Boltz's spatial/contiguous cropper.
    max_tokens, max_atoms : crop size (with a cropper); also pad targets when
        ``pad_to_max_*`` is set. Without a cropper, over-size structures are skipped.
    samples_per_epoch : required when ``sampler`` is not None.
    ids / exclude_ids : optional PDB-id allow/deny list (iterable or text-file path).
    filters : optional Boltz dynamic filters (objects with ``.filter(record) -> bool``)
        or plain callables ``record -> bool``.
    Symmetry features (``symmetry.pkl``) are not computed.
    """

    def __init__(
        self,
        target_dir: Union[str, Path],
        sampler: Union[None, str, Sampler] = None,
        cropper: Union[None, str, Cropper] = None,
        max_tokens: Optional[int] = 512,
        max_atoms: Optional[int] = 4608,
        samples_per_epoch: Optional[int] = None,
        manifest_path: Union[None, str, Path] = None,
        ids: Union[None, str, Path, Iterable[str]] = None,
        exclude_ids: Union[None, str, Path, Iterable[str]] = None,
        filters: Optional[list] = None,
        neighborhood_range: tuple[int, int] = (0, 40),
        pad_to_max_tokens: bool = False,
        pad_to_max_atoms: bool = False,
        atoms_per_window_queries: int = 32,
        min_dist: float = 2.0,
        max_dist: float = 22.0,
        num_bins: int = 64,
        binder_pocket_conditioned_prop: float = 0.0,
        binder_pocket_cutoff: float = 6.0,
        binder_pocket_sampling_geometric_p: float = 0.0,
        overfit: Optional[int] = None,
        seed: int = 42,
        max_retries: int = 20,
        return_id: bool = False,
    ) -> None:
        super().__init__()
        self.target_dir = Path(target_dir)
        self.max_tokens = max_tokens
        self.max_atoms = max_atoms
        self.pad_to_max_tokens = pad_to_max_tokens
        self.pad_to_max_atoms = pad_to_max_atoms
        self.atoms_per_window_queries = atoms_per_window_queries
        self.min_dist, self.max_dist, self.num_bins = min_dist, max_dist, num_bins
        self.pocket_prop = binder_pocket_conditioned_prop
        self.pocket_cutoff = binder_pocket_cutoff
        self.pocket_p = binder_pocket_sampling_geometric_p
        self.seed = seed
        self.max_retries = max_retries
        self.return_id = return_id
        self.tokenizer = BoltzTokenizer()

        # Records: manifest -> id filters -> dynamic filters -> overfit
        path = Path(manifest_path) if manifest_path else self.target_dir / "manifest.json"
        records = Manifest.load(path).records
        keep, drop = _read_ids(ids), _read_ids(exclude_ids)
        if keep is not None:
            records = [r for r in records if r.id.lower() in keep]
        if drop is not None:
            records = [r for r in records if r.id.lower() not in drop]
        for f in filters or []:
            fn: Callable[[Record], bool] = f.filter if hasattr(f, "filter") else f
            records = [r for r in records if fn(r)]
        if overfit is not None:
            records = records[:overfit]
        if not records:
            raise ValueError("No records left after filtering.")
        self.records = records

        # Sampler
        if sampler is None:
            self.sampler: Sampler = _SequentialSampler()
            self.sampled = False
        else:
            if samples_per_epoch is None:
                raise ValueError("samples_per_epoch is required when a sampler is used.")
            if isinstance(sampler, str):
                try:
                    sampler = {"random": RandomSampler, "cluster": ClusterSampler}[sampler]()
                except KeyError:
                    raise ValueError(f"Unknown sampler {sampler!r}") from None
            self.sampler = sampler
            self.sampled = True
        self.samples_per_epoch = samples_per_epoch

        # Cropper
        if isinstance(cropper, str):
            if cropper != "boltz":
                raise ValueError(f"Unknown cropper {cropper!r}")
            cropper = BoltzCropper(*neighborhood_range)
        self.cropper: Optional[Cropper] = cropper

        # Per-process state, created lazily so each DataLoader worker gets its own.
        self._rng: Optional[np.random.RandomState] = None
        self._iter: Optional[Iterator[Sample]] = None

    def __len__(self) -> int:
        return self.samples_per_epoch if self.sampled else len(self.records)

    # ------------------------------------------------------------------ internals
    def _init_worker_state(self) -> None:
        info = get_worker_info()
        wid = info.id if info else 0
        self._rng = np.random.RandomState((self.seed + 7919 * wid + torch.initial_seed()) % 2**32)
        if self.sampled:
            self._iter = self.sampler.sample(self.records, self._rng)

    def _next_sample(self, idx: int) -> Sample:
        if self.sampled:
            return next(self._iter)
        return Sample(record=_valid_only(self.records[idx]))

    def _featurize(self, sample: Sample) -> dict[str, Tensor]:
        record = sample.record
        structure = load_structure(record, self.target_dir)
        tokenized = self.tokenizer.tokenize(Input(structure, {}))

        if self.cropper is not None and self.max_tokens is not None:
            tokenized = self.cropper.crop(
                tokenized,
                max_tokens=self.max_tokens,
                random=self._rng,
                max_atoms=self.max_atoms,
                chain_id=sample.chain_id,
                interface_id=sample.interface_id,
            )
        if len(tokenized.tokens) == 0:
            raise ValueError("No tokens after cropping.")

        # Without a cropper, over-size structures cannot be padded; skip them.
        n_tok = len(tokenized.tokens)
        n_atoms = int(tokenized.tokens["atom_num"].sum())
        if self.cropper is None:
            if self.max_tokens is not None and n_tok > self.max_tokens:
                raise ValueError(f"{n_tok} tokens > max_tokens={self.max_tokens}")
            if self.max_atoms is not None and n_atoms > self.max_atoms:
                raise ValueError(f"{n_atoms} atoms > max_atoms={self.max_atoms}")

        pad_t = self.max_tokens if self.pad_to_max_tokens else None
        pad_a = self.max_atoms if self.pad_to_max_atoms else None
        feats = process_token_features(
            tokenized, pad_t, self.pocket_prop, self.pocket_cutoff, self.pocket_p
        )
        feats.update(
            process_atom_features(
                tokenized,
                self.atoms_per_window_queries,
                self.min_dist,
                self.max_dist,
                self.num_bins,
                pad_a,
                pad_t,
            )
        )
        if self.return_id:
            feats["record_id"] = record.id
        return feats

    def __getitem__(self, idx: int) -> dict[str, Tensor]:
        if self._rng is None:
            self._init_worker_state()
        # Boltz's featurizer uses the global numpy RNG for pocket conditioning.
        np.random.seed(self._rng.randint(2**31 - 1))

        last_err: Optional[Exception] = None
        for attempt in range(self.max_retries):
            # Non-sampled mode: on failure move to the next record.
            sample = self._next_sample((idx + attempt) % len(self.records))
            try:
                return self._featurize(sample)
            except Exception as e:  # noqa: BLE001
                last_err = e
                print(f"[loader] skipping {sample.record.id}: {e}")
        raise RuntimeError(f"Failed {self.max_retries} times in a row; last: {last_err}")


def collate(data: list[dict]) -> dict:
    """Stack features, padding to the batch max when shapes differ (as Boltz does)."""
    out = {}
    for key in data[0]:
        values = [d[key] for d in data]
        if key == "record_id":
            out[key] = values
            continue
        if key not in _NO_PAD_KEYS:
            if not all(v.shape == values[0].shape for v in values):
                values, _ = pad_to_max(values, 0)
            else:
                values = torch.stack(values, dim=0)
        out[key] = values
    return out


def build_dataloader(dataset: StructureSequenceDataset, batch_size: int = 1,
                     num_workers: int = 4, pin_memory: bool = True,
                     shuffle: Optional[bool] = None) -> DataLoader:
    """DataLoader helper. Shuffles only in non-sampled mode (sampling is already random)."""
    if shuffle is None:
        shuffle = not dataset.sampled
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers, pin_memory=pin_memory, collate_fn=collate)
