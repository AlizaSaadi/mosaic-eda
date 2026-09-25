"""Stratified sampling: keep each folder's share of the files, with a minimum per folder."""

from __future__ import annotations

import random
from collections import defaultdict

from mosaic.ingest.models import FileEntry, Modality, SampleSet


def allocate(sizes: dict[str, int], total: int, min_per_group: int) -> dict[str, int]:
    """Split `total` slots across groups in proportion to size (largest remainder),
    giving every group at least min(min_per_group, its size)."""
    available = sum(sizes.values())
    if available <= total:
        return dict(sizes)
    floor = {g: min(n, min_per_group) for g, n in sizes.items()}
    remaining = total - sum(floor.values())
    if remaining <= 0:
        # More groups than slots: take the floors from the largest groups first
        out, left = {}, total
        for g in sorted(sizes, key=lambda g: (-sizes[g], g)):
            take = min(floor[g], left)
            out[g] = take
            left -= take
        return out
    spare = {g: sizes[g] - floor[g] for g in sizes}
    spare_total = sum(spare.values())
    exact = {g: remaining * spare[g] / spare_total for g in sizes}
    out = {g: floor[g] + int(exact[g]) for g in sizes}
    leftover = total - sum(out.values())
    for g in sorted(sizes, key=lambda g: (-(exact[g] - int(exact[g])), g))[:leftover]:
        out[g] += 1
    return out


def stratified_sample(
    files: list[FileEntry],
    modality: Modality,
    max_files: int,
    *,
    seed: int = 42,
    min_per_group: int = 5,
) -> SampleSet:
    by_group: dict[str, list[FileEntry]] = defaultdict(list)
    for f in sorted(files, key=lambda f: f.path):
        by_group[f.group].append(f)
    quota = allocate({g: len(v) for g, v in by_group.items()}, max_files, min_per_group)
    rng = random.Random(seed)
    chosen: list[FileEntry] = []
    for group in sorted(by_group):
        members = by_group[group]
        take = quota.get(group, 0)
        chosen.extend(members if take >= len(members) else rng.sample(members, take))
    chosen.sort(key=lambda f: f.path)
    return SampleSet(
        modality=modality,
        files=chosen,
        total_available=len(files),
        per_group={g: n for g, n in quota.items() if n},
        seed=seed,
    )
