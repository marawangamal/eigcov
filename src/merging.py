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
    vectors: Sequence[_TaskVector], merge: str, *args, **kwargs
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
        keys = casted[0].lazy_keys()
        all_key_sets = [set(v.lazy_keys()) for v in casted]

        ignore_keys = kwargs.pop("ignore_keys", None) or []

        for key in tqdm(keys, desc="Merging task vectors", leave=False):
            if any(key not in ks for ks in all_key_sets):
                # Skip keys that are not present in all vectors
                continue
            # Stack on the merge device
            # NOTE: use get_vector_element to speedup lazy mode with caching
            taus = torch.stack([v.get_vector_element(key).to(device) for v in casted])

            if (
                taus[0].ndim == 2
                and "text_projection" not in key
                and max(taus[0].shape) < 20_000
                # and not in ignore_keys
                and not (ignore_keys and any(ik in key for ik in ignore_keys))
            ):
                # Only matrices can be merged using the merge function
                print(f"[{key}] merging layer with shape {taus[0].shape}")
                merged = merge_fn(taus, key=key, vectors=vectors, **kwargs)
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
def merge_isoc(taus: torch.Tensor, mode="mean", **kwargs):
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


merge_knots_isoc = lambda *args, **kwargs: _merge_knots(
    *args, merge_fn=merge_isoc, **kwargs
)
merge_knots_tsv = lambda *args, **kwargs: _merge_knots(
    *args, merge_fn=merge_tsv, **kwargs
)


pinv = torch.linalg.pinv


# ---------------------------------------------------------------------------
# RegMean
# ---------------------------------------------------------------------------


def merge_regmean(
    tau: torch.Tensor,
    key: str,
    vectors: Sequence[_TaskVector],
    scale_coef=None,
    max_dim=None,  # default to mean for dim(x) > max_dim
    sample_cov=False,  # sample covariance instead of population covariance
    **kwargs,
):
    c = []
    # km = v.param_key_to_cov_key(key)
    for v in vectors:
        km = v.param_key_to_cov_key(key)
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


# ---------------------------------------------------------------------------
# Fisher Merging
# ---------------------------------------------------------------------------
def _dinv(x, thresh=1e-8):
    return torch.where(x.abs() > thresh, 1 / x, 0)


def merge_fisher(
    tau: torch.Tensor,
    key: str,
    vectors: Sequence[_TaskVector],
    **kwargs,
):
    N, Do, Di = tau.shape
    f = []
    # km = v.param_key_to_cov_key(key)
    for v in vectors:
        km = v.param_key_to_cov_key(key)
        fpath = v.fisher_path
        if fpath is None:
            raise ValueError(f"No fisher path provided for task vector {v}")
        with np.load(fpath) as fdict:
            if km not in fdict:
                print(f"[skipped] {km} not found in {fpath}")
                return tau.mean(dim=0)
            f.append(fdict[km])

    # Shape: (N, Do*Di)
    f = torch.stack(
        [torch.as_tensor(x.reshape(-1), device=tau.device, dtype=tau.dtype) for x in f]
    )
    return _dinv(f.sum(dim=0)) * (f * tau.reshape(N, Do * Di)).sum(dim=0)


# ---------------------------------------------------------------------------
# Eigenvalue Covariance (EigCov)
# ---------------------------------------------------------------------------
def merge_eigcov(d: torch.Tensor, *args, **kwargs):
    c = d.transpose(1, 2) @ d
    return (d @ c).sum(dim=0) @ pinv(c.sum(dim=0))


def merge_eigcov_general(
    d: torch.Tensor,
    lam=0.0,
    alpha_weighted=False,
    cov_weighted=False,
    solver="lstsq",
    max_cond=42,
    **kwargs,
):
    # # DEBUG:
    # print(f"lam: {lam}, alpha_weighted: {alpha_weighted}, cov_weighted: {cov_weighted}")

    T, Do, Di = d.shape

    if cov_weighted:
        _c = d.transpose(1, 2) @ d  # (T, Di, Di)
        # c = c / (torch.linalg.norm(c, ord="fro", dim=(-2, -1), keepdim=True) ** 2)
        d = d / (torch.linalg.norm(_c, ord="fro", dim=(-2, -1), keepdim=True))

    # Factor each C_t = L_t L_t^T
    # e, v = torch.linalg.eigh(c)  # e: (T, Di), v: (T, Di, Di)
    # e = e.clamp(min=1e-8)
    # L = v * e.sqrt().unsqueeze(-2)  # (T, Di, Di)
    L = d
    Lt = L.transpose(-2, -1)  # (T, Di, Di)

    # Scale by sqrt(alpha_t)
    Lt_scaled = Lt
    if alpha_weighted:
        sqrt_alpha = 1.0 / d.flatten(1).norm(dim=1)  # (T,)
        Lt_scaled = sqrt_alpha[:, None, None] * Lt  # (T, Di, Di)

    A = Lt_scaled.reshape(T * Di, Di)  # (T*Di, Di)
    B = (Lt_scaled @ d.transpose(-1, -2)).reshape(T * Di, Do)  # (T*Di, Do)

    # Ridge regularization
    if lam > 0:
        A = torch.cat([A, lam**0.5 * torch.eye(Di, Di, device=A.device)], dim=0)
        B = torch.cat([B, torch.zeros(Di, Do, device=B.device)], dim=0)

    print(f"d.shape: {d.shape}, A.shape: {A.shape}, B.shape: {B.shape}")
    # Solve A @ X = B where X = W^T
    try:
        if solver == "lstsq":
            result = torch.linalg.lstsq(A, B)
            W = result.solution.T  # (Do, Di)
        elif solver == "solve":
            W = torch.linalg.solve(A.transpose(-2, -1) @ A, A.transpose(-2, -1) @ B).T
        return W
    except Exception as e:
        print(f"[fallback] solve failed ({e}), using mean")
        return d.mean(dim=0)


def merge_eigcov_gd(
    d: torch.Tensor,
    lam=0.0,
    alpha_weighted=False,
    cov_weighted=False,
    lr=1e-3,
    max_iters=1000,
    thresh=1e-6,
    **kwargs,
) -> torch.Tensor:
    """Gradient-descent solver for the EigCov objective.

    Minimizes the same weighted least-squares loss as merge_eigcov_general:
        L(W) = Σ_t tr((W - d_t) C_t (W - d_t)^T) + λ ‖W‖_F²
    where C_t = d_t^T @ d_t.
    """
    C = d.transpose(1, 2) @ d  # (T, Di, Di)

    if cov_weighted:
        C = C / (torch.linalg.norm(C, ord="fro", dim=(-2, -1), keepdim=True) ** 2)

    if alpha_weighted:
        alpha = 1.0 / d.flatten(1).norm(dim=1)  # (T,)
        C = alpha[:, None, None] * C  # (T, Di, Di)

    W = d.mean(dim=0).clone().requires_grad_(True)  # (Do, Di)
    optimizer = torch.optim.Adam([W], lr=lr)

    # Re-enable gradients inside the torch.no_grad() context of combine_task_vectors
    with torch.enable_grad():
        prev_loss = float("inf")
        for i in range(int(max_iters)):
            optimizer.zero_grad()
            diff = W.unsqueeze(0) - d  # (T, Do, Di)
            loss = (diff @ C).mul_(diff).sum()
            if lam > 0:
                loss = loss + lam * W.square().sum()
            loss.backward()
            optimizer.step()

            cur_loss = loss.item()
            if abs(prev_loss - cur_loss) / (abs(prev_loss) + 1e-12) < thresh:
                print(f"[converged] loss={cur_loss:.1e} < {thresh:.1e}")
                break
            prev_loss = cur_loss

    if i == int(max_iters) - 1:
        print(f"[not converged] loss={cur_loss:.1e} after {max_iters} iters")

    return W.detach()
