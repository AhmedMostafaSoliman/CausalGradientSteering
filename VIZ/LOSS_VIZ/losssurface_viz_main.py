import argparse
import os
import sys
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np

import matplotlib
matplotlib.use("Agg")


def _repo_root() -> Path:
    # This file lives under VIZ/LOSS_VIZ/, so repo root is two levels up
    return Path(__file__).resolve().parents[2]


def _ensure_repo_on_path():
    root = _repo_root()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))


_ensure_repo_on_path()


def _import_runtime_modules():
    import torch
    from domainbed import algorithms as db_algorithms
    from domainbed import datasets as db_datasets
    from domainbed.lib import misc as db_misc
    from domainbed.lib.fast_data_loader import FastDataLoader

    # Local sibling import for plotting helpers
    viz_dir = Path(__file__).parent
    if str(viz_dir) not in sys.path:
        sys.path.insert(0, str(viz_dir))
    import losssurface_plot as lsp

    return torch, db_algorithms, db_datasets, db_misc, FastDataLoader, lsp


def _params_to_vector(parameters) -> "np.ndarray":
    import torch
    return torch.cat([p.detach().view(-1) for p in parameters])


def _copy_flat_params_into(model, flat_params):
    import torch
    offset = 0
    with torch.no_grad():
        for p in model.parameters():
            sz = p.numel()
            p.copy_(flat_params[offset:offset + sz].view_as(p))
            offset += sz


def _get_basis(w1, w2, w3):
    """Compute orthonormal basis spanning the plane defined by (w1,w2,w3).

    Returns unit vectors u,v and magnitudes du,dv such that:
      w2 ≈ w1 + du * u,  w3 ≈ w1 + dv * v (with u ⟂ v)
    """
    import torch
    u = w2 - w1
    du = u.norm()
    u = u / (du + 1e-12)

    v = w3 - w1
    v = v - torch.dot(u, v) * u
    dv = v.norm()
    v = v / (dv + 1e-12)
    return u, v, du, dv


def _get_xy(point, origin, u, v):
    import torch
    d = point - origin
    return torch.as_tensor([torch.dot(d, u), torch.dot(d, v)])


def _eval_loss_acc(model, loader, device) -> Tuple[float, float]:
    import torch
    import torch.nn.functional as F
    model.eval()
    n = 0
    correct = 0.0
    loss_sum = 0.0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            logits = model.predict(x) if hasattr(model, "predict") else model(x)
            loss = F.cross_entropy(logits, y, reduction="sum")
            loss_sum += loss.item()
            if logits.size(1) == 1:
                preds = (logits > 0).long().view(-1)
            else:
                preds = logits.argmax(dim=1)
            correct += (preds == y).float().sum().item()
            n += y.numel()
    acc = correct / max(n, 1)
    mean_loss = loss_sum / max(n, 1)
    return mean_loss, acc


def _build_eval_loaders_from_ckpt(
    ckpt: dict,
    env_index: int,
    batch_size: int,
    max_samples: Optional[int],
    which: str = "both",
):
    """Recreate dataset + eval loader(s) for a given environment's in/out splits.

    which: 'in' | 'out' | 'both'
    Returns a dict like { 'in': loader or None, 'out': loader or None }
    """
    _, _, db_datasets, db_misc, FastDataLoader, _ = _import_runtime_modules()

    dataset_name = ckpt["args"]["dataset"]
    data_dir = ckpt["args"]["data_dir"]
    test_envs = ckpt["args"].get("test_envs", [])
    hparams = ckpt["model_hparams"]
    holdout_fraction = ckpt["args"].get("holdout_fraction", 0.2)
    trial_seed = ckpt["args"].get("trial_seed", 0)

    ds_class = db_datasets.get_dataset_class(dataset_name)
    mds = ds_class(data_dir, test_envs, hparams)

    # Build in/out split for the selected environment
    env = mds.datasets[env_index]
    out, in_ = db_misc.split_dataset(env, int(len(env) * holdout_fraction), trial_seed)

    def make_loader(dset):
        if max_samples and len(dset) > max_samples:
            dset, _ = db_misc.split_dataset(dset, max_samples, trial_seed)
        return FastDataLoader(dataset=dset, batch_size=batch_size, num_workers=mds.N_WORKERS)

    loaders = {"in": None, "out": None}
    if which in ("in", "both"):
        loaders["in"] = make_loader(in_)
    if which in ("out", "both"):
        loaders["out"] = make_loader(out)
    return loaders


def _build_train_test_loaders_from_ckpt(
    ckpt: dict,
    test_env_index: int,
    batch_size: int,
    max_samples: Optional[int],
):
    """Build loaders for 'test' (selected env) and 'train' (all other envs).

    - 'test': in-split of the selected test_env_index
    - 'train': concatenation of in-splits from all other environments
    """
    _, _, db_datasets, db_misc, FastDataLoader, _ = _import_runtime_modules()
    import torch
    from torch.utils.data import ConcatDataset

    dataset_name = ckpt["args"]["dataset"]
    data_dir = ckpt["args"]["data_dir"]
    test_envs = ckpt["args"].get("test_envs", [])
    hparams = ckpt["model_hparams"]
    holdout_fraction = ckpt["args"].get("holdout_fraction", 0.2)
    trial_seed = ckpt["args"].get("trial_seed", 0)

    ds_class = db_datasets.get_dataset_class(dataset_name)
    mds = ds_class(data_dir, test_envs, hparams)

    in_splits = {}
    for env_i, env in enumerate(mds.datasets):
        out, in_ = db_misc.split_dataset(env, int(len(env) * holdout_fraction), trial_seed)
        in_splits[env_i] = in_

    test_dset = in_splits[test_env_index]
    train_dsets = [in_splits[i] for i in range(len(mds.datasets)) if i != test_env_index]

    # Concatenate train envs
    if len(train_dsets) == 1:
        train_cat = train_dsets[0]
    else:
        train_cat = ConcatDataset(train_dsets)

    # Optionally subsample
    def maybe_sample(dset):
        if max_samples and len(dset) > max_samples:
            dset, _ = db_misc.split_dataset(dset, max_samples, trial_seed)
        return dset

    test_dset = maybe_sample(test_dset)
    train_cat = maybe_sample(train_cat)

    loaders = {
        "test": FastDataLoader(dataset=test_dset, batch_size=batch_size, num_workers=mds.N_WORKERS),
        "train": FastDataLoader(dataset=train_cat, batch_size=batch_size, num_workers=mds.N_WORKERS),
    }
    return loaders


def _load_algorithm_from_ckpt(ckpt_path: str, device):
    import torch
    _, db_algorithms, _, _, _, _ = _import_runtime_modules()

    ckpt = torch.load(ckpt_path, map_location="cpu")
    algo_name = ckpt["args"]["algorithm"]
    AlgoClass = db_algorithms.get_algorithm_class(algo_name)
    algo = AlgoClass(
        ckpt["model_input_shape"],
        ckpt["model_num_classes"],
        ckpt["model_num_domains"],
        ckpt["model_hparams"],
    )
    algo.load_state_dict(ckpt["model_dict"], strict=True)
    algo.to(device).eval()
    return algo, ckpt


def _resolve_model_pkls(path: str) -> List[str]:
    p = Path(path)
    if p.is_file() and p.name == "model.pkl":
        return [str(p)]
    if p.is_dir():
        # Search recursively for model.pkl within run directories
        return [str(pp) for pp in p.rglob("model.pkl")]
    raise FileNotFoundError(f"No model.pkl found at {path}")


def _pick_three_models(paths: List[str]) -> List[str]:
    if len(paths) >= 3:
        return paths[:3]
    return paths


def _make_random_plane_around(w1, eps_scale: float = 0.1, seed: int = 0):
    """Return (w1, w2, w3) where w2,w3 are random perturbations around w1.
    eps_scale is a fraction of ||w1|| used as step size.
    """
    import torch
    g = torch.Generator(device=w1.device)
    g.manual_seed(seed)
    n = w1.numel()
    base_norm = w1.norm() + 1e-12
    step = eps_scale * base_norm
    r1 = torch.randn(n, generator=g, device=w1.device)
    r2 = torch.randn(n, generator=g, device=w1.device)
    r1 = r1 / (r1.norm() + 1e-12)
    r2 = r2 - torch.dot(r1, r2) * r1
    r2 = r2 / (r2.norm() + 1e-12)
    w2 = w1 + step * r1
    w3 = w1 + step * r2
    return w1, w2, w3


def run(
    source: str,
    test_env: Optional[int] = None,
    grid: Optional[int] = None,
    margin: float = 0.6,
    batch_size: int = 128,
    max_samples: int = 3000,
    out_dir: str = "VIZ/LOSS_VIZ/output",
    log_alpha: float = -5.0,
    seed: int = 0,
    plane_mode: str = "random",  # 'random' or 'three'
    eps_scale: float = 0.1,
    eval_splits: str = "both",  # legacy: 'in' | 'out' | 'both' (for env-splits)
    eval_mode: str = "train_test",  # 'train_test' | 'envsplits'
    point_only: bool = True,
):
    """
    Visualize loss surface around a trained model.

    Args:
      source: Path to a model.pkl, a run directory, or a parent directory containing multiple runs.
      test_env: Optional explicit env index; if None, will read from checkpoint args.
      grid: Number of ticks per axis; if None, set to int((1+2*margin)*15).
      margin: Horizontal/vertical margin around the [0,1] range.
      batch_size: Eval batch size.
      max_samples: Max samples from the eval split to speed up.
      out_dir: Where to write results and plots.
      log_alpha: Log-scale alpha parameter for contour plotting.
      seed: RNG seed used for random-plane fallback.
    """
    torch, _, _, _, FastDataLoader, lsp = _import_runtime_modules()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Resolve model checkpoints
    model_paths = _resolve_model_pkls(source)
    sel_models = _pick_three_models(model_paths)
    if len(sel_models) == 0:
        raise FileNotFoundError(f"No model.pkl found under {source}")

    # Load first model for meta
    model1, ckpt1 = _load_algorithm_from_ckpt(sel_models[0], device)
    te = test_env if test_env is not None else (ckpt1["args"].get("test_envs", [0])[0])

    # Build eval loader(s)
    if eval_mode == "envsplits":
        loaders_env = _build_eval_loaders_from_ckpt(ckpt1, te, batch_size, max_samples, which=eval_splits)
        to_eval = [("in", loaders_env.get("in")), ("out", loaders_env.get("out"))]
    else:
        loaders_tt = _build_train_test_loaders_from_ckpt(ckpt1, te, batch_size, max_samples)
        to_eval = [("test", loaders_tt["test"]), ("train", loaders_tt["train"])]

    # Prepare weight vectors
    w1 = _params_to_vector(model1.parameters()).to(device)
    if plane_mode == "three" and len(sel_models) >= 3:
        model2, _ = _load_algorithm_from_ckpt(sel_models[1], device)
        model3, _ = _load_algorithm_from_ckpt(sel_models[2], device)
        w2 = _params_to_vector(model2.parameters()).to(device)
        w3 = _params_to_vector(model3.parameters()).to(device)
    else:
        # Random plane around w1 (single-checkpoint mode)
        w1, w2, w3 = _make_random_plane_around(w1, eps_scale=eps_scale, seed=seed)

    # Basis for the plane
    u, v, du, dv = _get_basis(w1, w2, w3)

    # Grid definition
    if grid is None:
        grid = int((1.0 + 2 * margin) * 15)
    alphas = np.linspace(0.0 - margin, 1.0 + margin, grid)
    betas = np.linspace(0.0 - margin, 1.0 + margin, grid)

    # Header rows: (w1,w2,w3) positions in the plane
    header_rows = []
    for i, w in enumerate([w1, w2, w3]):
        c = _get_xy(w, w1, u, v)
        header_rows.append({"ij": f"w{i+1}", "grid_xy": c})

    # Work on a deepcopy of model1 to avoid mutating it
    base_model = type(model1)(
        ckpt1["model_input_shape"],
        ckpt1["model_num_classes"],
        ckpt1["model_num_domains"],
        ckpt1["model_hparams"],
    ).to(device).eval()
    base_model.load_state_dict(model1.state_dict(), strict=True)

    # Evaluate the grid for a given split
    def _eval_for_loader(loader):
        results = list(header_rows)
        tk_total = grid * grid
        idx = 0
        for i, alpha in enumerate(alphas):
            for j, beta in enumerate(betas):
                if (idx % 25) == 0:
                    print(f"  [{idx:5d}/{tk_total}] i={i+1}/{grid}, j={j+1}/{grid}")
                interpolated = w1 + alpha * du * u + beta * dv * v
                _copy_flat_params_into(base_model, interpolated)
                mean_loss, acc = _eval_loss_acc(base_model, loader, device)
                results.append({
                    "ij": [i, j],
                    "grid_xy": torch.as_tensor([alpha * du, beta * dv]),
                    "error": 1.0 - float(acc),
                    "loss": float(mean_loss),
                })
                idx += 1
        return results

    # Prepare output path
    out_dir = Path(out_dir)
    # Compose a unique name from parent of run dir (if present)
    try:
        parent_stem = Path(sel_models[0]).parents[1].stem  # e.g., hyperparam folder
    except Exception:
        parent_stem = "run"
    runopt = f"G{grid}_margin{margin}_[{0}-{grid//2}-{grid-1}]"
    out_dir_final = out_dir / parent_stem
    out_dir_final.mkdir(parents=True, exist_ok=True)

    # Evaluate requested split(s) and plot
    import os as _os
    for name, loader in to_eval:
        if loader is None:
            continue
        print(f"Evaluating [{name}] ...")
        results_here = _eval_for_loader(loader)
        # Use simpler names when in train_test mode
        if eval_mode == "train_test":
            out_path = out_dir_final / f"TE{te}_{name}.pth"
        else:
            out_path = out_dir_final / f"TE{te}_env{te}_{name}.pth"
        torch.save(results_here, out_path)
        print(f"Saved raw results: {out_path}")
        try:
            lsp.setup_matplotlib()
            _os.environ["LOSSVIZ_POINT_ONLY"] = "1" if point_only else "0"
            lsp.plot_from_path(out_path, log_alpha=log_alpha)
            print(f"Saved contour plots next to: {out_path}")
        except Exception as e:
            print(f"Plotting failed ({name}): {e}")


def main():
    parser = argparse.ArgumentParser(description="Loss surface visualization around a model.pkl")
    parser.add_argument("--source", required=True, help="Path to model.pkl or a directory containing runs")
    parser.add_argument("--test_env", type=int, default=None, help="Test environment index (overrides checkpoint)")
    parser.add_argument("--grid", type=int, default=None, help="Grid resolution per axis (None -> auto)")
    parser.add_argument("--margin", type=float, default=0.6, help="Margin around [0,1] along axes")
    parser.add_argument("--batch_size", type=int, default=128, help="Eval batch size")
    parser.add_argument("--max_samples", type=int, default=3000, help="Max eval samples for speed")
    parser.add_argument("--out_dir", type=str, default="VIZ/LOSS_VIZ/output", help="Output directory for results and plots")
    parser.add_argument("--log_alpha", type=float, default=-5.0, help="Log-scale alpha for contour plotting")
    parser.add_argument("--seed", type=int, default=0, help="Seed for random-plane generation")
    parser.add_argument("--plane_mode", type=str, choices=["random", "three"], default="random", help="Plane definition: random around checkpoint or through three checkpoints")
    parser.add_argument("--eps_scale", type=float, default=0.1, help="Step size as fraction of ||w|| for random plane")
    parser.add_argument("--eval_splits", type=str, choices=["in", "out", "both"], default="both", help="Env-split mode: which split(s) to evaluate for the selected env")
    parser.add_argument("--eval_mode", type=str, choices=["train_test", "envsplits"], default="train_test", help="Evaluate on train/test distributions (train=all non-test envs), or per-env in/out splits")
    parser.add_argument("--show_triangles", action="store_true", help="Plot three weight points and center instead of only checkpoint")
    args = parser.parse_args()

    run(
        source=args.source,
        test_env=args.test_env,
        grid=args.grid,
        margin=args.margin,
        batch_size=args.batch_size,
        max_samples=args.max_samples,
        out_dir=args.out_dir,
        log_alpha=args.log_alpha,
        seed=args.seed,
        plane_mode=args.plane_mode,
        eps_scale=args.eps_scale,
        eval_splits=args.eval_splits,
        eval_mode=args.eval_mode,
        point_only=(not args.show_triangles),
    )


if __name__ == "__main__":
    main()
