"""General-purpose online covariance estimation via forward hooks.

Works with any PyTorch model.  Vision-specific module types (e.g.
MultiHeadAttentionPacked) can be passed via ``extra_module_types``.
"""

import torch


class OnlineCovariance:
    def __init__(self, dim1, dim2=1, device="cpu", mode="cov"):
        self.device = device
        self.meanx = torch.zeros((dim1, dim2), device=device)
        self.meany = torch.zeros((dim1, dim2), device=device)
        self.C = torch.zeros((dim1, dim1), device=device)
        self.n = 0
        self.add = {
            "cov": self._add_cov,
            "sm": self._add_second_moment,
        }[mode]

    @property
    def cov(self):
        # Population covariance
        return self.C / self.n

    @property
    def cov_sample(self):
        # Sample covariance
        return self.C / (self.n - 1)

    def _add_cov(self, x, y):
        x = x.to(self.device)
        y = y.to(self.device)
        self.n += 1
        dx = x - self.meanx
        self.meanx += dx / self.n
        self.meany += (y - self.meany) / self.n
        self.C += dx @ (y - self.meany).T

    def _add_second_moment(self, x, y):
        x = x.to(self.device)
        y = y.to(self.device)
        self.n += 1
        # Uncentered second moment: E[X X^T]
        self.C += x @ y.T


def register_hooks(
    model,
    cov_device,
    cov_type="sm",
    cov_estimator="full",
    mask_ref=None,
    batch_first=False,
    extra_module_types=(),
):
    """Register forward hooks to collect per-layer covariance.

    Args:
        model: PyTorch model.
        cov_device: Device to store covariance matrices on.
        cov_type: Covariance mode passed to OnlineCovariance ("sm" or "cov").
        cov_estimator: "full" for full-sequence DxT vectors, "sampled" for a
            single random token position per sample.
        mask_ref: A list of mask tensors (each shape (B, T)) that the caller
            updates each batch before the forward pass. The hook matches the
            mask whose T dimension equals the activation's T. Pass None to
            disable masking. Only supported when batch_first=True.
        batch_first: If True, activations are (B, T, D); if False, (T, B, D).
            Vision (OpenCLIP) is sequence-first (False); language (T5/HF) is
            batch-first (True).
        extra_module_types: Additional module types to hook beyond
            nn.Linear and nn.MultiheadAttention (e.g. custom MHA variants).

    Returns:
        cobjs: dict mapping layer name → OnlineCovariance.
        handles: list of hook handles (call h.remove() when done).
    """
    base_types = (torch.nn.Linear, torch.nn.MultiheadAttention)
    hook_types = base_types + tuple(extra_module_types)

    cobjs = {}
    handles = []

    for name, module in model.named_modules():
        if not isinstance(module, hook_types):
            continue

        def make_hook(n):
            def hook(mod, inp, out):
                x = inp[0] if isinstance(inp, (tuple, list)) else inp
                if not isinstance(x, torch.Tensor):
                    return

                if batch_first:
                    B, T, D = x.shape
                    # Pick the mask whose sequence length matches T
                    mask = None
                    if mask_ref is not None:
                        for m in mask_ref:
                            if m is not None and m.shape[1] == T:
                                mask = m
                                break
                    if mask is not None:
                        x = x * mask.unsqueeze(-1).float()
                else:
                    T, B, D = x.shape

                if cov_estimator == "sampled":
                    # Dx1 vector: one random token position per sample
                    if n not in cobjs:
                        cobjs[n] = OnlineCovariance(
                            D, device=cov_device, mode=cov_type
                        )
                    cobj = cobjs[n]
                    for b in range(B):
                        j = torch.randint(0, T, (1,)).item()
                        if batch_first:
                            cobj.add(x[b, j : j + 1].T, x[b, j : j + 1].T)
                        else:
                            cobj.add(x[j : j + 1, b].T, x[j : j + 1, b].T)
                else:
                    # DxT vector: full sequence per sample
                    if n not in cobjs:
                        cobjs[n] = OnlineCovariance(
                            D, T, device=cov_device, mode=cov_type
                        )
                    cobj = cobjs[n]
                    for b in range(B):
                        if batch_first:
                            cobj.add(x[b, :].T, x[b, :].T)
                        else:
                            cobj.add(x[:, b].T, x[:, b].T)

            return hook

        handles.append(module.register_forward_hook(make_hook(name)))

    return cobjs, handles
