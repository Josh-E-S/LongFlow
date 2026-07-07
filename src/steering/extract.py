"""P0 Stage 2 — steering-vector extraction + diagnostics (spec: p0-steering.md Stage 2).

Consumes PairRecords from contrast_pairs.py. Per axis, per layer:
direction = mean(positive-pole activations) - mean(negative-pole activations).

Diagnostics run BEFORE any injection:
- consistency: mean pairwise cosine of per-script directions within an axis.
  If per-script directions don't agree, the "axis" is noise (expect >0.4 on
  mid layers per EmoSteer-adjacent findings).
- independence: per-layer cosine between arousal and valence directions.
  ~1.0 means one "expressiveness" axis, not two — that's still a result, log it.
- per-layer direction norms (pre-normalization) -> candidate injection layers
  (expect the middle third of the stack).

Runs on CPU; no GPU or vibevoice dependency.
"""

from dataclasses import dataclass

import torch

from src.steering.contrast_pairs import PairRecord, load_records


def _mean_by_script_pole(
    records: list[PairRecord], axis: str
) -> dict[str, dict[str, torch.Tensor]]:
    """{script_id: {pole: mean-over-samples [L, d]}} for one axis."""
    acc: dict[str, dict[str, list[torch.Tensor]]] = {}
    for r in records:
        if r.axis != axis:
            continue
        acc.setdefault(r.script_id, {}).setdefault(r.pole, []).append(r.layer_vectors)
    out: dict[str, dict[str, torch.Tensor]] = {}
    for sid, poles in acc.items():
        if set(poles) != {"pos", "neg"}:
            continue  # incomplete pair (e.g. one pole failed in the capture loop)
        out[sid] = {p: torch.stack(vs).mean(dim=0) for p, vs in poles.items()}
    if not out:
        raise ValueError(f"no complete pos/neg script pairs for axis {axis!r}")
    return out


def per_script_directions(records: list[PairRecord], axis: str) -> dict[str, torch.Tensor]:
    """Unnormalized per-script directions [L, d]: mean(pos) - mean(neg)."""
    return {
        sid: poles["pos"] - poles["neg"]
        for sid, poles in _mean_by_script_pole(records, axis).items()
    }


def consistency(script_dirs: dict[str, torch.Tensor]) -> torch.Tensor:
    """Per-layer mean pairwise cosine across scripts [L]."""
    dirs = torch.stack(list(script_dirs.values()))  # [S, L, d]
    unit = torch.nn.functional.normalize(dirs, dim=-1)
    sims = torch.einsum("sld,tld->lst", unit, unit)  # [L, S, S]
    s = unit.shape[0]
    if s < 2:
        raise ValueError("need >= 2 scripts for consistency")
    off_diag = sims.sum(dim=(1, 2)) - sims.diagonal(dim1=1, dim2=2).sum(dim=1)
    return off_diag / (s * (s - 1))


@dataclass
class AxisExtraction:
    axis: str
    direction: torch.Tensor  # [L, d], unit-normalized per layer
    norms: torch.Tensor  # [L], pre-normalization magnitude of the mean direction
    consistency: torch.Tensor  # [L]
    num_scripts: int


def extract_axis(records: list[PairRecord], axis: str) -> AxisExtraction:
    script_dirs = per_script_directions(records, axis)
    mean_dir = torch.stack(list(script_dirs.values())).mean(dim=0)  # [L, d]
    norms = mean_dir.norm(dim=-1)
    return AxisExtraction(
        axis=axis,
        direction=torch.nn.functional.normalize(mean_dir, dim=-1),
        norms=norms,
        consistency=consistency(script_dirs),
        num_scripts=len(script_dirs),
    )


def independence(a: AxisExtraction, b: AxisExtraction) -> torch.Tensor:
    """Per-layer cosine between two axis directions [L] (directions are unit)."""
    return (a.direction * b.direction).sum(dim=-1)


def candidate_layers(ext: AxisExtraction, top_k: int = 5) -> list[int]:
    """Layers ranked by consistency x norm — where the signal is both strong
    and script-agnostic. Injection candidates for Stage 3."""
    score = ext.consistency * ext.norms
    return score.argsort(descending=True)[:top_k].tolist()


def extract_all(records_path) -> dict:
    """Full Stage 2: load records, extract both axes, compute diagnostics.

    Returns a plain dict (torch.save-able) with everything Stage 3 needs.
    """
    records = load_records(records_path)
    axes = sorted({r.axis for r in records})
    extractions = {a: extract_axis(records, a) for a in axes}
    out = {
        "directions": {a: e.direction for a, e in extractions.items()},
        "norms": {a: e.norms for a, e in extractions.items()},
        "consistency": {a: e.consistency for a, e in extractions.items()},
        "num_scripts": {a: e.num_scripts for a, e in extractions.items()},
        "candidate_layers": {a: candidate_layers(e) for a, e in extractions.items()},
        "num_records": len(records),
    }
    if len(axes) == 2:
        out["independence"] = independence(*extractions.values())
    return out


def summarize(result: dict) -> str:
    """Human-readable diagnostic table for NOTES.md."""
    lines = [f"records: {result['num_records']}"]
    for axis, cons in result["consistency"].items():
        n_layers = len(cons)
        third = n_layers // 3
        mid = cons[third : 2 * third]
        lines.append(
            f"{axis}: scripts={result['num_scripts'][axis]}  "
            f"consistency mean={cons.mean():.3f} mid-third={mid.mean():.3f} "
            f"max={cons.max():.3f}@L{int(cons.argmax())}  "
            f"norm max@L{int(result['norms'][axis].argmax())}  "
            f"candidates={result['candidate_layers'][axis]}"
        )
    if "independence" in result:
        ind = result["independence"]
        lines.append(
            f"arousal-valence cosine: mean={ind.mean():.3f} "
            f"max={ind.max():.3f}@L{int(ind.abs().argmax())} "
            f"(~1.0 => single expressiveness axis)"
        )
    return "\n".join(lines)
