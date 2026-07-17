"""Repo-level reproduction wrapper for HZ0000/NNPI.

This script reproduces the public NN-only, 4-input communication-network path
from https://github.com/HZ0000/NNPI without modifying the cloned repository.

It intentionally does not reproduce the full paper tables: the public repo does
not include the M/M/1 code or the SK/SCP/QRF/SCQR baselines.
"""

from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
import tempfile
from pathlib import Path


def _parse_lambda_grid(value: str) -> list[int]:
    """Parse either "15:33" (Python range stop-exclusive) or "15,21,27,32"."""
    value = value.strip()
    if ":" in value:
        start_text, stop_text = value.split(":", 1)
        return list(range(int(start_text), int(stop_text)))
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def _ensure_jax_stub_and_reexec() -> None:
    """Mask an incompatible local jaxlib before TensorFlow/Keras import.

    On this machine, importing TensorFlow/Keras pulls in jaxlib, whose installed
    build requires unsupported AVX instructions. A tiny local jax stub is enough
    for the NNPI code path because TensorFlow Lite conversion is not used.
    """
    if os.environ.get("NNPI_REPRO_CHILD") == "1":
        return

    stub_root = Path(tempfile.gettempdir()) / "nnpi_tf_stub"
    jax_dir = stub_root / "jax"
    jax_dir.mkdir(parents=True, exist_ok=True)
    (jax_dir / "__init__.py").write_text(
        "def xla_computation(*args, **kwargs):\n"
        "    raise RuntimeError('Temporary jax stub for NNPI reproduction.')\n",
        encoding="utf-8",
    )

    env = os.environ.copy()
    env["NNPI_REPRO_CHILD"] = "1"
    env.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    env.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "nnpi_mpl_config"))
    env.setdefault("XDG_CACHE_HOME", str(Path(tempfile.gettempdir()) / "nnpi_xdg_cache"))
    old_pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = str(stub_root) if not old_pythonpath else f"{stub_root}{os.pathsep}{old_pythonpath}"
    os.execve(sys.executable, [sys.executable, *sys.argv], env)


def _load_nnpi_definitions(nnpi_dir: Path) -> dict:
    """Load NNPI model/loss/calibration definitions, but not its main run."""
    source_path = nnpi_dir / "pi_computer.py"
    code = source_path.read_text(encoding="utf-8")
    prefix = code.split("\nnum_re=3", 1)[0]

    # The public script was generated from Colab and assumes tensorflow.keras.
    # The local environment has standalone Keras available but tensorflow.keras
    # is broken, so patch imports in memory only.
    prefix = prefix.replace("import tensorflow.keras as keras", "import keras")
    prefix = prefix.replace("from tensorflow.keras.layers import Dense, Flatten", "from keras.layers import Dense, Flatten")
    prefix = prefix.replace("from tensorflow.keras import Model", "from keras import Model")
    prefix = prefix.replace("tf.keras.optimizers.Adam", "keras.optimizers.Adam")
    prefix = prefix.replace("tf.keras.models.load_model", "keras.models.load_model")

    namespace: dict = {"__name__": "__nnpi_repro__"}
    cwd = Path.cwd()
    os.chdir(nnpi_dir)
    try:
        exec(prefix, namespace)
    finally:
        os.chdir(cwd)
    return namespace


def _make_train_function(namespace: dict, epochs: int, seed: int):
    import keras
    import numpy as np
    import tensorflow as tf

    MyModel = namespace["MyModel"]
    train_step = namespace["train_step"]

    def train_network(x_train, y_train, penalty_lambda: int):
        optimizer = keras.optimizers.Adam(learning_rate=0.005)
        model = MyModel()
        train_ds = (
            tf.data.Dataset.from_tensor_slices((x_train, y_train))
            .shuffle(100, seed=seed, reshuffle_each_iteration=True)
            .batch(1)
        )
        loss_history: list[float] = []
        for epoch in range(epochs):
            epoch_losses = []
            for images, labels in train_ds:
                loss = train_step(images, labels, optimizer, model, penalty_lambda)
                epoch_losses.append(float(loss.numpy()))
            mean_loss = float(np.mean(epoch_losses))
            loss_history.append(mean_loss)
            print(f"lambda={penalty_lambda} epoch={epoch + 1}/{epochs} loss={mean_loss:.6f}", flush=True)
        return model, loss_history

    return train_network


def _compute_candidate_predictions(models: dict[int, object], lambdas: list[int], x):
    import numpy as np

    lower_list = []
    upper_list = []
    for penalty_lambda in lambdas:
        pred = models[penalty_lambda].predict(x, verbose=0)
        lower_list.append(np.min(pred, axis=-1))
        upper_list.append(np.max(pred, axis=-1))
    return np.asarray(lower_list), np.asarray(upper_list)


def _validation_arrays(datatr, slice_index: int, numval: int, val_reps: int):
    import numpy as np

    x_val_tmp = datatr[1 + 3 * numval :, (5 * slice_index + 1) : (5 * slice_index + 5)]
    y_val_tmp = (datatr[1 + 3 * numval :, 5 * slice_index + 5] * 1000).reshape([-1, 1])
    x_val = np.zeros(shape=x_val_tmp.shape)
    y_val = np.zeros(shape=y_val_tmp.shape)
    for site_idx in range(numval):
        for rep_idx in range(val_reps):
            x_val[site_idx * val_reps + rep_idx] = x_val_tmp[rep_idx * numval + site_idx]
            y_val[site_idx * val_reps + rep_idx] = y_val_tmp[rep_idx * numval + site_idx]
    return x_val, y_val


def _calibrate(method: str, val_lower, val_upper, test_lower, test_upper, y_val, y_test, numval: int, val_reps: int, rng):
    """Replicate the NNVA/NNGN/NNGU model-selection rules from pi_computer.py."""
    import numpy as np

    target = 0.95
    conf_level = 0.95
    n_models = val_lower.shape[0]
    margin = 0.0

    W = np.zeros((n_models, numval))
    for model_idx in range(n_models):
        for site_idx in range(numval):
            for rep_idx in range(val_reps):
                flat_idx = site_idx * val_reps + rep_idx
                if val_lower[model_idx, flat_idx] - margin <= y_val[flat_idx] <= val_upper[model_idx, flat_idx] + margin:
                    W[model_idx, site_idx] += 1.0 / val_reps

    mean_site_cov = np.mean(W, axis=-1)
    val_cov = np.mean((val_lower - margin <= y_val.reshape(1, -1)) & (y_val.reshape(1, -1) <= val_upper + margin), axis=1)
    test_cov = np.mean((test_lower - margin <= y_test.reshape(1, -1)) & (y_test.reshape(1, -1) <= test_upper + margin), axis=1)
    val_width = np.mean(val_upper - val_lower, axis=1)
    test_width = np.mean(test_upper - test_lower, axis=1)

    if method == "NNVA":
        feasible = val_cov > target
        criterion_cov = val_cov
    else:
        cov_matrix = np.cov(W)
        if np.ndim(cov_matrix) == 0:
            cov_matrix = np.asarray([[float(cov_matrix)]])
        sigma = np.sqrt(np.maximum(np.diag(cov_matrix), 0.0)) + 0.001
        samples = rng.multivariate_normal(np.zeros(n_models), cov_matrix, size=10000, check_valid="ignore")
        if method == "NNGN":
            gaussian_max = np.max(samples / sigma.reshape(1, -1), axis=1)
            q = float(np.quantile(gaussian_max, conf_level))
            threshold = target + q * sigma / np.sqrt(numval)
        elif method == "NNGU":
            gaussian_max = np.max(samples, axis=1)
            q = float(np.quantile(gaussian_max, conf_level))
            threshold = target + q / np.sqrt(numval)
        else:
            raise ValueError(f"Unknown method: {method}")
        feasible = mean_site_cov > threshold
        criterion_cov = mean_site_cov

    if np.any(feasible):
        feasible_indices = np.where(feasible)[0]
        selected_local = int(np.argmin(val_width[feasible_indices]))
        selected = int(feasible_indices[selected_local])
    else:
        selected = int(np.argmax(criterion_cov))

    return {
        "selected_index": selected,
        "selected_val_cov": float(val_cov[selected]),
        "selected_test_cov": float(test_cov[selected]),
        "selected_mean_site_val_cov": float(mean_site_cov[selected]),
        "selected_val_width": float(val_width[selected]),
        "selected_test_width": float(test_width[selected]),
        "n_feasible": int(np.sum(feasible)),
    }


def _plot_summary(summary_csv: Path, output_path: Path) -> None:
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    df = pd.read_csv(summary_csv)
    x = np.arange(len(df))
    colors = ["#B279A2", "#F58518", "#E45756"]
    fig, ax = plt.subplots(figsize=(6.8, 4.3), dpi=160)
    ax.bar(x, df["EP"], color=colors[: len(df)], alpha=0.85)
    ax.axhline(0.95, color="#333333", linestyle="--", linewidth=1.1)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("EP")
    ax.set_xticks(x)
    ax.set_xticklabels(df["method"])
    ax.set_title("NNPI repo-level reproduction summary")
    ax.grid(axis="y", color="#E0E0E0", linewidth=0.55)
    ax2 = ax.twinx()
    ax2.plot(x, df["IW"], color="#222222", marker="o", linewidth=1.4)
    ax2.set_ylabel("IW")
    for i, row in df.iterrows():
        ax.text(i, row["EP"] + 0.02, f"cov {row['mean_test_cov']:.3f}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def run(args: argparse.Namespace) -> None:
    import numpy as np
    import tensorflow as tf

    rng = np.random.default_rng(args.seed)
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    nnpi_dir = Path(args.nnpi_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    namespace = _load_nnpi_definitions(nnpi_dir)
    datatr = namespace["datatr"]
    datate = namespace["datate"]
    namespace["numval"] = args.numval

    lambdas = _parse_lambda_grid(args.lambdas)
    train_network = _make_train_function(namespace, args.epochs, args.seed)

    x_train = datatr[1 : (1 + args.train_reps * args.numval), 1:5]
    y_train = (datatr[1 : (1 + args.train_reps * args.numval), 5] * 1000).reshape([-1, 1])
    x_test = datate[1:, 1:5]
    y_test = (datate[1:, 5] * 1000).reshape([-1, 1])

    print(f"nnpi_dir={nnpi_dir}")
    print(f"output_dir={output_dir}")
    print(f"train_shape={x_train.shape}, test_shape={x_test.shape}")
    print(f"lambdas={lambdas}, epochs={args.epochs}, n_slices={args.n_slices}")

    models = {}
    training_rows = []
    for penalty_lambda in lambdas:
        model, loss_history = train_network(x_train, y_train, penalty_lambda)
        models[penalty_lambda] = model
        for epoch_idx, loss_value in enumerate(loss_history, start=1):
            training_rows.append({"lambda": penalty_lambda, "epoch": epoch_idx, "loss": loss_value})

    training_csv = output_dir / "training_loss.csv"
    with training_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["lambda", "epoch", "loss"])
        writer.writeheader()
        writer.writerows(training_rows)

    print("precomputing test predictions")
    test_lower, test_upper = _compute_candidate_predictions(models, lambdas, x_test)

    per_slice_rows = []
    for slice_index in range(args.n_slices):
        print(f"validation slice {slice_index + 1}/{args.n_slices}", flush=True)
        x_val, y_val = _validation_arrays(datatr, slice_index, args.numval, args.val_reps)
        val_lower, val_upper = _compute_candidate_predictions(models, lambdas, x_val)
        for method in ["NNVA", "NNGN", "NNGU"]:
            result = _calibrate(method, val_lower, val_upper, test_lower, test_upper, y_val, y_test, args.numval, args.val_reps, rng)
            selected_lambda = lambdas[result["selected_index"]]
            row = {
                "method": method,
                "slice": slice_index,
                "selected_lambda": selected_lambda,
                **{key: value for key, value in result.items() if key != "selected_index"},
            }
            per_slice_rows.append(row)
            print(
                f"  {method}: lambda={selected_lambda}, "
                f"test_cov={row['selected_test_cov']:.6f}, "
                f"test_width={row['selected_test_width']:.6f}, "
                f"n_feasible={row['n_feasible']}",
                flush=True,
            )

    per_slice_csv = output_dir / "per_slice_results.csv"
    fieldnames = [
        "method",
        "slice",
        "selected_lambda",
        "selected_val_cov",
        "selected_test_cov",
        "selected_mean_site_val_cov",
        "selected_val_width",
        "selected_test_width",
        "n_feasible",
    ]
    with per_slice_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(per_slice_rows)

    summary_rows = []
    for method in ["NNVA", "NNGN", "NNGU"]:
        rows = [row for row in per_slice_rows if row["method"] == method]
        test_cov = np.asarray([row["selected_test_cov"] for row in rows], dtype=float)
        test_width = np.asarray([row["selected_test_width"] for row in rows], dtype=float)
        val_cov = np.asarray([row["selected_val_cov"] for row in rows], dtype=float)
        summary_rows.append(
            {
                "method": method,
                "n_slices": len(rows),
                "EP": float(np.mean(test_cov > 0.95)),
                "IW": float(np.mean(test_width)),
                "mean_test_cov": float(np.mean(test_cov)),
                "sd_test_cov": float(np.std(test_cov, ddof=1)) if len(rows) > 1 else 0.0,
                "mean_val_cov": float(np.mean(val_cov)),
                "mean_val_width": float(np.mean([row["selected_val_width"] for row in rows])),
            }
        )

    summary_csv = output_dir / "summary_results.csv"
    with summary_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    if args.plot:
        _plot_summary(summary_csv, output_dir / "summary_results.png")

    print("wrote:")
    for path in [training_csv, per_slice_csv, summary_csv, output_dir / "summary_results.png"]:
        if path.exists():
            print(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nnpi-dir", default="/private/tmp/NNPI", help="Path to a clone of HZ0000/NNPI.")
    parser.add_argument("--output-dir", default="analysis/nnpi_repro/output_repo_level", help="Directory for CSV/plot outputs.")
    parser.add_argument("--lambdas", default="15:33", help='Penalty lambda grid, e.g. "15:33" or "15,21,27,32".')
    parser.add_argument("--epochs", type=int, default=30, help="Training epochs per lambda.")
    parser.add_argument("--n-slices", type=int, default=50, help="Number of validation slices/macro repetitions.")
    parser.add_argument("--numval", type=int, default=100, help="Number of validation sites per slice.")
    parser.add_argument("--train-reps", type=int, default=3, help="Training replications used from truth_4input_tr2.csv.")
    parser.add_argument("--val-reps", type=int, default=2, help="Validation replications used from truth_4input_tr2.csv.")
    parser.add_argument("--seed", type=int, default=20260629)
    parser.add_argument("--plot", action="store_true", help="Write summary_results.png.")
    parser.add_argument(
        "--profile",
        choices=["full", "smoke"],
        default="full",
        help="Use full defaults or a fast smoke-test profile.",
    )
    args = parser.parse_args()

    if args.profile == "smoke":
        args.lambdas = "15,21,27,32"
        args.epochs = min(args.epochs, 2)
        args.n_slices = min(args.n_slices, 3)

    _ensure_jax_stub_and_reexec()
    run(args)


if __name__ == "__main__":
    main()
