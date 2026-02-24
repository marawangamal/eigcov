from argparse import Namespace
from pyexpat import model
import sys
import time
from pathlib import Path
import torch
from typing import Callable, Sequence, Union
from tqdm import tqdm
import numpy as np

from src.task_vectors import _TaskVector


# Type: (key, [w1, w2, ...]) -> merged tensor
TensorMergeFn = Callable[[str, Sequence[torch.Tensor]], torch.Tensor]


def combine_task_vectors(
    vectors: Sequence[_TaskVector], merge: str, args: Namespace
) -> _TaskVector:
    """Generic combiner for task vectors.

    Args:
        vectors: list of task vectors (same logical type)
        merge: name of the function defined in this module to merge the task vectors

    Returns:
        A new task vector with the merged task vectors.
    """
    vectors = list(vectors)
    assert len(vectors) > 0, "Need at least one task vector"

    # Get the function (must be defined in this module)
    merge_fn = getattr(sys.modules[__name__], "merge_" + merge)

    # Prefer GPU for merging if available; results are moved back to CPU so they
    # stay compatible with the rest of the pipeline.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    base = vectors[0]
    # Cast others to same type as base
    casted = [base] + [base._cast_to_same_type(v) for v in vectors[1:]]

    start_time = time.time()

    with torch.no_grad():
        new_vector = {}
        for key in tqdm(
            casted[0].vector,
            desc="Merging task vectors",
            total=len(casted[0].vector),
            leave=False,
        ):
            if any(key not in v.vector for v in casted):
                # Skip keys that are not present in all vectors
                continue
            # Stack on the merge device
            # NOTE: use get_vector_element to speedup lazy mode with caching
            taus = torch.stack([v.get_vector_element(key).to(device) for v in casted])

            if (
                taus[0].ndim == 2
                and "text_projection" not in key
                and max(taus[0].shape) < 10_000
            ):
                # Only matrices can be merged using the merge function
                merged = merge_fn(taus, key=key, vectors=vectors)
            else:
                # For all other tensors, we average the values
                merged = taus.mean(dim=0)

            # Keep merged parameters on CPU for compatibility with checkpoint loading.
            # i.e. for when we do `task_vector.apply_to(pretrained_checkpoint, scaling_coef=1.0)`
            new_vector[key] = merged.to("cpu")

    print(f"Merging task vectors took {(time.time() - start_time) / 60:.2f} minutes")
    return base.__class__(vector=new_vector)


# ---------------------------------------------------------------------------
# Basic
# ---------------------------------------------------------------------------


def merge_sum(taus: torch.Tensor, **kwargs) -> torch.Tensor:
    # Shape: (N, Do, Di) -> (Do, Di)
    return taus.sum(dim=0)


def merge_mean(taus: torch.Tensor, **kwargs) -> torch.Tensor:
    return taus.mean(dim=0)


# ---------------------------------------------------------------------------
# TSV
# ---------------------------------------------------------------------------


def _compute_procrustes(x: torch.Tensor) -> torch.Tensor:
    u, _, vt = torch.linalg.svd(x, full_matrices=False)
    return u @ vt


def merge_tsv(taus: torch.Tensor, **kwargs) -> torch.Tensor:
    """Computes the TSV merge of the given tensors.

    Computes: Uo  Dc Vto

    Args:
        tensors (torch.Tensor): The tensors to merge. Shape: (N_tasks, Di, Do)

    Returns:
        torch.Tensor: The merged tensors. Shape: (Di, Do)
    """

    N_tasks = len(taus)
    u, s, vt = torch.linalg.svd(taus, full_matrices=False)
    R = min(u.shape[1], vt.shape[2])
    Rp = R // N_tasks
    u, s, vt = u[:, :, :Rp], s[:, :Rp], vt[:, :Rp, :]

    # # # w/o decorrelation
    # tau_bl = torch.einsum("bij,bj,bjk->bik", u, s, vt)
    # tau[layer_name] = tau_bl.sum(dim=0)

    # w/ decorrelation
    B, Di, _ = u.shape
    _, _, Do = vt.shape
    # (Di, B, R)
    u_hat = u.permute(1, 0, 2).reshape(Di, B * Rp)
    s_hat = s.reshape(-1)
    vt_hat = vt.reshape(B * Rp, Do)
    u_ortho = _compute_procrustes(u_hat)  # (Di, Rp)
    vt_ortho = _compute_procrustes(vt_hat.T).T  # (Rp, Do)
    tau_l = torch.einsum("ij,j,jk->ik", u_ortho, s_hat, vt_ortho)  # (Di, Do)
    return tau_l


# ---------------------------------------------------------------------------
# ISOC
# ---------------------------------------------------------------------------
def _merge_isoc(taus: torch.Tensor, mode="mean", **kwargs):
    m = taus.sum(dim=0)
    u, s, vt = torch.linalg.svd(m, full_matrices=False)
    if mode == "mean":
        s_iso = s.mean() * torch.ones_like(s)
    elif mode == "unity":
        s_iso = torch.ones_like(s)
    elif mode == "rms":
        s_iso = torch.sqrt((s**2).mean()) * torch.ones_like(s)
    elif mode == "spectral":
        s_iso = s[0] * torch.ones_like(s)  # Use largest singular value
    else:
        raise ValueError(f"Unknown mode: {mode}")
    return torch.einsum("ik,k,kj->ij", u, s_iso, vt)


merge_isoc_mean = lambda *args, **kwargs: _merge_isoc(*args, mode="mean", **kwargs)
merge_isoc_rms = lambda *args, **kwargs: _merge_isoc(*args, mode="rms", **kwargs)


# ---------------------------------------------------------------------------
# KNOTS
# ---------------------------------------------------------------------------
def _merge_knots(taus: torch.Tensor, merge_fn: Callable, **kwargs) -> torch.Tensor:
    # print device
    N, Do, Di = taus.shape
    d = taus.permute(1, 0, 2).reshape(Do, N * Di)
    # [Do, R], [R, R], [R, N*Di]
    u, s, vt = torch.linalg.svd(d, full_matrices=False)
    # taus_tilde = torch.einsum("ij,jnk->nik", torch.diag(s), vt.reshape(-1, N, Di))
    # tau_tilde = merge_fn(taus_tilde, **kwargs)
    # return u @ tau_tilde
    # [R, N*Di] -> [N, R, Di] -> [R, Di]
    tau_tilde = merge_fn(vt.reshape(-1, N, Di).permute(1, 0, 2), **kwargs)
    return torch.einsum("or,r,ri->oi", u, s, tau_tilde)


merge_knots_ta = lambda *args, **kwargs: _merge_knots(
    *args, merge_fn=merge_isoc_mean, **kwargs
)
merge_knots_isoc_mean = lambda *args, **kwargs: _merge_knots(
    *args, merge_fn=merge_isoc_rms, **kwargs
)
merge_knots_isoc_rms = lambda *args, **kwargs: _merge_knots(
    *args, merge_fn=merge_isoc_rms, **kwargs
)
merge_knots_tsv = lambda *args, **kwargs: _merge_knots(
    *args, merge_fn=merge_tsv, **kwargs
)

merge_knots_eigcov = lambda *args, **kwargs: _merge_knots(
    *args, merge_fn=merge_eigcov, **kwargs
)


# ---------------------------------------------------------------------------
# Subspace Alignment (SA)
# ---------------------------------------------------------------------------
def merge_sa(taus: torch.Tensor, *args, **kwargs):
    m = torch.bmm(taus, taus.transpose(1, 2)).sum(dim=0)  # memory intensive
    # m = torch.einsum("nik,njk->ij", taus, taus)
    # check if nans/infs
    if torch.isnan(m).any() or torch.isinf(m).any():
        raise ValueError("m contains nans or infs")
    u, s, vt = torch.linalg.svd(m, full_matrices=False)
    q = u @ vt
    return (q @ taus).mean(dim=0)


pinv = torch.linalg.pinv


# ---------------------------------------------------------------------------
# RegMean
# ---------------------------------------------------------------------------
def _param_key_to_module_key(key: str):
    return "image_encoder." + key.replace(".weight", "")


# Test normalized accuracy: 0.8934755825227096
# Test absolute accuracy: 0.6888413285902395
def _merge_regmean(
    tau: torch.Tensor,
    key: str,
    vectors: Sequence[_TaskVector],
    scale_coef=None,
    max_dim=None,  # default to mean for dim(x) > max_dim
    sample_cov=False,  # sample covariance instead of population covariance
    **kwargs,
):
    c = []
    km = _param_key_to_module_key(key)
    for v in vectors:
        cpath = v.covariance_path
        if cpath is None:
            raise ValueError(f"No covariance path provided for task vector {v}")
        with np.load(cpath) as cdict:
            if km not in cdict:
                print(f"[skipped] {km} not found in {cpath}")
                return tau.mean(dim=0)
            ct = cdict[km]
            if max_dim is not None and ct.shape[1] > max_dim:
                print(f"[skipped] {km} has shape {cdict[km].shape} > {max_dim}")
                return tau.mean(dim=0)
            if sample_cov:
                ct = (cdict[f"{km}_n"] * ct) / (cdict[f"{km}_n"] - 1)
            c.append(ct)
    c = torch.stack([torch.as_tensor(x, device=tau.device, dtype=tau.dtype) for x in c])

    if scale_coef is not None:
        m_diag = (
            torch.eye(c.shape[1], device=c.device, dtype=c.dtype)
            .unsqueeze(0)
            .expand(c.shape[0], -1, -1)
        )
        c = scale_coef * c + (1 - scale_coef) * m_diag * c
    return (tau @ c).sum(dim=0) @ pinv(c.sum(dim=0))


merge_regmean = lambda *args, **kwargs: _merge_regmean(*args, **kwargs)
merge_regmean_09 = lambda *args, **kwargs: _merge_regmean(
    *args, scale_coef=0.9, **kwargs
)
merge_regmean_08 = lambda *args, **kwargs: _merge_regmean(
    *args, scale_coef=0.8, **kwargs
)
merge_regmean_mx1000 = lambda *args, **kwargs: _merge_regmean(
    *args, max_dim=1000, **kwargs
)
merge_regmean_sc = lambda *args, **kwargs: _merge_regmean(
    *args, sample_cov=True, **kwargs
)


# ---------------------------------------------------------------------------
# Eigenvalue Covariance (EigCov)
# ---------------------------------------------------------------------------
def _get_eigcov(d: torch.Tensor, *args, **kwargs):
    c = d.transpose(1, 2) @ d
    return c


def merge_eigcov(d: torch.Tensor, *args, **kwargs):
    # c = d.transpose(1, 2) @ d
    c = _get_eigcov(d)
    return (d @ c).sum(dim=0) @ pinv(c.sum(dim=0))


# ---------------------------------------------------------------------------
# Projected RegMean
# ---------------------------------------------------------------------------


def _loss_prm(w_star, q, w, c):
    """Compute projected regmean loss.

    .. math::
        sum_t Tr[(w_star - q_t w_t) c (w_star - q_t w_t)^T]

    Args:
        w_star: (Do, Di) - consensus matrix
        q: (N, Do, Do) - orthogonal matrices
        w: (N, Do, Di) - input matrices
        c: (N, Di, Di) - covariance matrix

    Returns:
        loss: scalar - sum_t tr[(w_star - q_t w_t) sigma (w_star - q_t w_t)^T]
    """
    residuals = w_star.unsqueeze(0) - q @ w  # (N, Do, Di)
    # Tr[(w_star - q_t w_t) c (w_star - q_t w_t)^T]
    return torch.einsum("noi,nij,njo->", residuals, c, residuals.transpose(1, 2))


merge_prm = lambda *args, **kwargs: _merge_prm(*args, **kwargs)
merge_prm_eigcov = lambda *args, **kwargs: _merge_prm(
    *args, c_train=_get_eigcov(*args), c_test=_get_eigcov(*args), **kwargs
)


def _merge_prm(
    taus, c_train=None, c_test=None, max_iter=100, tol=1e-3, key=None, **kwargs
):
    """Solve: min_{w_con, q_t ortho} sum_t tr[(w_con - q_t w_t) sigma (w_con - q_t w_t)^T]

    Args:
        w: (N, Do, Di) - input matrices
        c: (N, Di, Di) - covariance matrix (defaults to identity)
        max_iter: maximum alternating minimization iterations
        tol: convergence tolerance on loss change

    Returns:
        w_star: (Do, Di) - consensus matrix
        q: (N, Do, Do) - orthogonal matrices
        losses: list of scalar loss values per iteration
    """
    N, Do, Di = taus.shape

    # Only orthogonal invariant layers are considered by prm
    if key is None or not "qk.weight" in key or not "vot.weight" in key:
        if c_train is None:
            return taus.mean(dim=0)
        else:
            return (taus @ c_train).sum(dim=0) @ pinv(c_train.sum(dim=0))

    if c_train is None:
        c_train = (
            torch.eye(Di, device=taus.device, dtype=taus.dtype)
            .unsqueeze(0)
            .expand(N, -1, -1)
        )
    if c_test is None:
        c_test = (
            torch.eye(Di, device=taus.device, dtype=taus.dtype)
            .unsqueeze(0)
            .expand(N, -1, -1)
        )

    w_star = taus.mean(dim=0)
    losses = []

    c_sum_pinv = torch.linalg.pinv(c_train.sum(dim=0))
    pbar = tqdm(range(max_iter), desc="Projected RegMean iterations", leave=False)

    for iteration in pbar:
        # Update q mats: orthogonal Procrustes
        u, _, vt = torch.linalg.svd(
            w_star @ c_train @ taus.transpose(1, 2), full_matrices=False
        )
        q_star = u @ vt  # (N, Do, Di)

        # Update w_star: mean of rotated matrices
        w_star = (q_star @ taus @ c_train).sum(dim=0) @ c_sum_pinv

        # Compute loss
        loss = _loss_prm(w_star, q_star, taus, c_test)
        losses.append(loss.item())
        pbar.set_postfix(loss=losses[-1])
        if iteration > 0 and abs(losses[-2] - losses[-1]) < tol:
            break

    return w_star


# ---------------------------------------------------------------------------
# Double-sided Projected RegMean
# ---------------------------------------------------------------------------
def _procrustes(m):
    u, _, vt = torch.linalg.svd(m, full_matrices=False)
    return u @ vt


def _loss_dprm(w_star, q, p, w):
    residuals = w_star.unsqueeze(0) - q @ w @ p.transpose(1, 2)  # (N, Do, Di)
    # Tr[(w_star - q_t w_t p_t.T) I (w_star - q_t w_t p_t.T)^T]
    return torch.einsum("noi,nio->", residuals, residuals.transpose(1, 2))


def merge_dprm(taus, max_iter=100, tol=1e-3, key=None, **kwargs):
    """Double-sided Projected RegMean.

    Args:
        taus: (N, Do, Di) - input matrices
        max_iter: maximum alternating minimization iterations
        tol: convergence tolerance on loss change
    """
    N, Do, Di = taus.shape

    # Only orthogonal invariant layers are considered by prm
    if key is None or not "qk.weight" in key or not "vot.weight" in key:
        return taus.mean(dim=0)

    w_star = taus.mean(dim=0)
    losses = []

    pbar = tqdm(range(max_iter), desc="Projected RegMean iterations", leave=False)
    q_star = (
        torch.eye(Do, device=taus.device, dtype=taus.dtype)
        .unsqueeze(0)
        .expand(N, -1, -1)
    )
    p_star = (
        torch.eye(Do, device=taus.device, dtype=taus.dtype)
        .unsqueeze(0)
        .expand(N, -1, -1)
    )

    for iteration in pbar:
        # Update q mats: orthogonal Procrustes
        q_star = _procrustes(w_star @ p_star @ taus.transpose(1, 2))
        p_star = _procrustes(w_star.transpose(1, 2) @ q_star @ taus)
        w_star = (q_star @ taus @ p_star.transpose(1, 2)).sum(dim=0)

        # Compute loss
        loss = _loss_dprm(w_star, q_star, p_star, taus)
        losses.append(loss.item())
        pbar.set_postfix(loss=losses[-1])
        if iteration > 0 and abs(losses[-2] - losses[-1]) < tol:
            break

    return w_star
