#!/usr/bin/env python3
"""Deep-network helpers for the revision analyses (plan B4, B10, B13, B14, B15).

EEGNet-v4 and ShallowFBCSPNet from braindecode, trained with Adam (lr 1e-3),
cross-entropy, batch 64, at most `epochs` epochs, early stopping on validation
accuracy with patience 6. Validation split modes:
  'trial'   - 15 % stratified slice of the training trials (used for the
              trial-random and run-disjoint protocols, consistent with them);
  'subject' - 15 % of the training SUBJECTS (GroupShuffleSplit), disjoint from
              both the remaining training subjects and the test subjects
              (nested subject-level validation for subject-disjoint protocols);
  'none'    - fixed epoch budget, no early stopping (within-subject training on
              ~30 trials, plan B13).
Preprocessing modes: 'ztrial' (per-trial per-channel z-score, as submitted) or
'global' (one scalar scale = 1/median|x| of the training fold, plan B15).
"""

from __future__ import annotations

import inspect
import hashlib

import numpy as np
import torch

DEFAULT_EPOCHS = 40
PATIENCE = 6
BATCH = 64
LR = 1e-3
VAL_FRAC = 0.15


def zscore(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32, copy=False)
    mean = x.mean(axis=2, keepdims=True)
    std = x.std(axis=2, keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    return (x - mean) / std


def global_scale_factor(x_train: np.ndarray) -> float:
    med = float(np.median(np.abs(x_train)))
    return 1.0 / med if med > 0 else 1.0


def preprocess(x: np.ndarray, mode: str, scale: float | None = None) -> np.ndarray:
    if mode == "ztrial":
        return zscore(x)
    if mode == "global":
        return (x.astype(np.float32, copy=False) * np.float32(scale)).astype(np.float32)
    raise ValueError(mode)


def _filter_kwargs(cls, kwargs: dict) -> dict:
    sig = inspect.signature(cls.__init__).parameters
    return {k: v for k, v in kwargs.items() if k in sig}


def make_model(name: str, n_chans: int, n_times: int, n_outputs: int = 2, width: int | None = None):
    from braindecode.models import ShallowFBCSPNet

    try:  # braindecode <= 1.7
        from braindecode.models import EEGNetv4 as EEGNetCls
    except ImportError:  # braindecode >= 1.8 renamed the class
        from braindecode.models import EEGNet as EEGNetCls

    if name == "eegnet_v4":
        kw = {"n_chans": n_chans, "n_outputs": n_outputs, "n_times": n_times}
        if width is not None:
            kw.update({"F1": int(width), "D": 2, "F2": int(2 * width)})
        fk = _filter_kwargs(EEGNetCls, kw)
        if width is not None and not {"F1", "D", "F2"} <= set(fk):
            raise RuntimeError(f"EEGNet width parameters not accepted by {EEGNetCls.__name__}: {list(fk)}")
        return EEGNetCls(**fk)
    if name == "shallow_fbcsp":
        kw = {"n_chans": n_chans, "n_outputs": n_outputs, "n_times": n_times, "final_conv_length": "auto"}
        if width is not None:
            kw.update({"n_filters_time": int(width), "n_filters_spat": int(width)})
        fk = _filter_kwargs(ShallowFBCSPNet, kw)
        if width is not None and not {"n_filters_time", "n_filters_spat"} <= set(fk):
            raise RuntimeError("ShallowFBCSPNet width parameters not accepted")
        return ShallowFBCSPNet(**fk)
    raise ValueError(name)


def count_params(model) -> int:
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


def _predict(model, x: torch.Tensor, batch: int = 512) -> np.ndarray:
    model.eval()
    out = []
    with torch.no_grad():
        for s in range(0, len(x), batch):
            out.append(model(x[s : s + batch]).argmax(1).cpu().numpy())
    return np.concatenate(out) if out else np.empty(0, dtype=np.int64)


def explicit_validation_split(n_train: int, validation_indices: np.ndarray):
    """Validate train-local held-out IDs and return disjoint fit/validation IDs."""
    idx_val = np.asarray(validation_indices)
    if idx_val.ndim != 1 or not np.issubdtype(idx_val.dtype, np.integer):
        raise ValueError("validation_indices must be one-dimensional integer indices")
    idx_val = idx_val.astype(np.int64, copy=False)
    if not len(idx_val) or len(np.unique(idx_val)) != len(idx_val):
        raise ValueError("validation_indices must be nonempty and unique")
    if np.any(idx_val < 0) or np.any(idx_val >= n_train):
        raise ValueError("validation_indices are outside the training array")
    idx_fit = np.setdiff1d(np.arange(n_train), idx_val)
    if not len(idx_fit):
        raise ValueError("explicit validation leaves no optimization trials")
    assert not np.intersect1d(idx_fit, idx_val).size
    assert len(idx_fit) + len(idx_val) == n_train
    return idx_fit, idx_val


def index_sha256(indices: np.ndarray) -> str:
    """Hash a global trial-index set in sorted little-endian int64 order."""
    values = np.sort(np.asarray(indices, dtype=np.int64)).astype("<i8", copy=False)
    return hashlib.sha256(values.tobytes()).hexdigest()


def train_predict(
    model_name: str,
    x_tr: np.ndarray,
    y_tr: np.ndarray,
    x_te: np.ndarray,
    device: torch.device,
    seed: int,
    *,
    val_mode: str = "trial",
    groups_tr: np.ndarray | None = None,
    epochs: int = DEFAULT_EPOCHS,
    patience: int = PATIENCE,
    batch_size: int = BATCH,
    lr: float = LR,
    preproc: str = "ztrial",
    width: int | None = None,
    return_model: bool = False,
    validation_indices: np.ndarray | None = None,
) -> dict:
    """Train and predict raw trials; explicit validation IDs are train-local."""
    from sklearn.model_selection import GroupShuffleSplit, train_test_split

    torch.manual_seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)
    n_out = int(max(2, len(np.unique(y_tr))))
    scale = global_scale_factor(x_tr) if preproc == "global" else None
    x_tr_p = preprocess(x_tr, preproc, scale)
    x_te_p = preprocess(x_te, preproc, scale)

    if validation_indices is not None:
        idx_fit, idx_val = explicit_validation_split(len(y_tr), validation_indices)
    elif val_mode == "trial":
        idx_fit, idx_val = train_test_split(np.arange(len(y_tr)), test_size=VAL_FRAC, random_state=seed, stratify=y_tr)
    elif val_mode == "subject":
        if groups_tr is None:
            raise ValueError("val_mode='subject' needs groups_tr")
        gss = GroupShuffleSplit(n_splits=1, test_size=VAL_FRAC, random_state=seed)
        idx_fit, idx_val = next(gss.split(np.zeros(len(y_tr)), y_tr, groups_tr))
    elif val_mode == "none":
        idx_fit, idx_val = np.arange(len(y_tr)), np.empty(0, dtype=np.int64)
    else:
        raise ValueError(val_mode)

    xt = torch.from_numpy(x_tr_p[idx_fit]).to(device)
    yt = torch.from_numpy(y_tr[idx_fit]).to(device)
    xv = torch.from_numpy(x_tr_p[idx_val]).to(device) if len(idx_val) else None
    yv = y_tr[idx_val]
    model = make_model(model_name, x_tr.shape[1], x_tr.shape[2], n_out, width=width).to(device)
    n_params = count_params(model)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    crit = torch.nn.CrossEntropyLoss()
    best_state, best_acc, bad, best_epoch, epochs_run = None, -1.0, 0, 0, 0
    for epoch in range(epochs):
        model.train()
        order = torch.from_numpy(rng.permutation(len(xt))).to(device)
        for s in range(0, len(order), batch_size):
            b = order[s : s + batch_size]
            opt.zero_grad()
            loss = crit(model(xt[b]), yt[b])
            loss.backward()
            opt.step()
        epochs_run = epoch + 1
        if xv is None:
            continue
        acc = float(np.mean(_predict(model, xv) == yv))
        if acc > best_acc:
            best_acc, bad, best_epoch = acc, 0, epoch + 1
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    x_all_tr = torch.from_numpy(x_tr_p).to(device)
    train_acc = float(np.mean(_predict(model, x_all_tr) == y_tr))
    preds = _predict(model, torch.from_numpy(x_te_p).to(device))
    out = {
        "preds": preds,
        "train_acc": train_acc,
        "val_acc": float(best_acc) if xv is not None else float("nan"),
        "best_epoch": int(best_epoch) if xv is not None else int(epochs_run),
        "epochs_run": int(epochs_run),
        "n_params": n_params,
        "n_fit": int(len(idx_fit)),
        "n_val": int(len(idx_val)),
    }
    if return_model:
        out["model"] = model
        out["scale"] = scale
    return out


def fixed_test_fits(data: dict, model_name: str, device, seed: int, *, epochs: int,
                    preproc: str = "ztrial", width: int | None = None):
    """Yield E17 rows, holding the exact BG validation trials fixed across arms."""
    import common as C

    if preproc != "ztrial":
        raise ValueError("E17 fixed-test networks require per-trial z-scoring (plan B21/B4)")
    x, y, subject = data["x"], data["y"], data["subject"]
    for rotation in range(3):
        design = C.fixed_test_arms(data, data["dataset"], seed, rotation)
        te = np.asarray(design["test"], dtype=np.int64)
        bg = np.asarray(design["bg"], dtype=np.int64)
        validation = np.asarray(design["validation"], dtype=np.int64)
        assert len(validation) and len(np.unique(validation)) == len(validation)
        assert np.isin(validation, bg).all(), "validation must be selected solely from BG"
        assert not np.intersect1d(validation, te).size
        validation_hash = index_sha256(validation)
        bg_hash = index_sha256(bg)
        expected_validation = None
        for arm in ("A0", "A1", "A2"):
            tr = np.asarray(design["arms"][arm], dtype=np.int64)
            assert len(np.unique(tr)) == len(tr)
            assert not np.intersect1d(tr, te).size
            assert np.isin(bg, tr).all()
            assert np.isin(validation, tr).all()
            inverse = np.full(len(y), -1, dtype=np.int64)
            inverse[tr] = np.arange(len(tr))
            val_local = inverse[validation]
            fit_local, val_local = explicit_validation_split(len(tr), val_local)
            actual_validation = tr[val_local]
            if expected_validation is None:
                expected_validation = actual_validation.copy()
            assert np.array_equal(actual_validation, expected_validation)
            assert index_sha256(actual_validation) == validation_hash
            assert not np.intersect1d(tr[fit_local], validation).size
            r = train_predict(model_name, x[tr], y[tr], x[te], device, seed,
                              val_mode="trial", groups_tr=subject[tr], epochs=epochs,
                              preproc=preproc, width=width, validation_indices=val_local)
            assert r["n_fit"] == len(fit_local) and r["n_val"] == len(validation)
            base = {"dataset": data["dataset"], "task": data["task"], "model": model_name,
                    "preproc": preproc, "protocol": "fixed_test", "reference": "bg_fixed",
                    "arm": arm, "seed": seed, "rotation": rotation}
            if data["dataset"] == "eegbci":
                base["width"] = width if width is not None else "default"
            ps_rows = []
            for s in np.unique(subject[te]):
                mask = subject[te] == s
                ps_rows.append({**base, "subject": int(s), "n_test": int(mask.sum()),
                                "accuracy": C.fmt(np.mean(r["preds"][mask] == y[te][mask]), 12)})
            pf_row = {**base, "fold": rotation, "n_train": len(tr), "n_test": len(te),
                      "n_train_subjects": len(np.unique(subject[tr])),
                      "n_class0": int(np.sum(y[tr] == 0)), "n_class1": int(np.sum(y[tr] == 1)),
                      "n_fit": r["n_fit"], "n_val": r["n_val"],
                      "train_acc": C.fmt(r["train_acc"], 12), "val_acc": C.fmt(r["val_acc"], 12),
                      "test_acc": C.fmt(np.mean(r["preds"] == y[te]), 12),
                      "best_epoch": r["best_epoch"], "selected_epoch": r["best_epoch"],
                      "epochs_run": r["epochs_run"], "n_params": r["n_params"],
                      "training_seed": seed, "validation_source": "bg_stratified_15pct",
                      "validation_sha256": validation_hash, "bg_sha256": bg_hash,
                      "test_sha256": index_sha256(te), "train_sha256": index_sha256(tr),
                      "fit_sha256": index_sha256(tr[fit_local]),
                      "leftover_subject": design["leftover_subject"]}
            print(f"    {model_name:14s} fixed_test seed={seed} rotation={rotation} arm={arm} "
                  f"test_acc={pf_row['test_acc']} selected_epoch={r['best_epoch']}", flush=True)
            yield ps_rows, pf_row


def fine_tune_predict(model, x_cal: np.ndarray, y_cal: np.ndarray, x_te: np.ndarray, device, seed: int, *, epochs: int = 20, lr: float = LR, preproc: str = "ztrial", scale: float | None = None) -> np.ndarray:
    """Fine-tune a trained network on k labelled calibration trials, then predict (plan B14)."""
    import copy

    torch.manual_seed(seed)
    m = copy.deepcopy(model)
    if len(y_cal) == 0:
        return _predict(m, torch.from_numpy(preprocess(x_te, preproc, scale)).to(device))
    xt = torch.from_numpy(preprocess(x_cal, preproc, scale)).to(device)
    yt = torch.from_numpy(y_cal).to(device)
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    crit = torch.nn.CrossEntropyLoss()
    m.train()
    for _ in range(epochs):
        opt.zero_grad()
        loss = crit(m(xt), yt)
        loss.backward()
        opt.step()
    return _predict(m, torch.from_numpy(preprocess(x_te, preproc, scale)).to(device))
