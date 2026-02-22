import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import re
import unittest


class MultiHeadAttentionSplit(nn.Module):
    def __init__(self, d_model, n_head, bias=False):
        super().__init__()
        assert d_model % n_head == 0
        self.d_model = d_model
        self.n_head = n_head
        self.d_k = d_model // n_head
        self.q = nn.Linear(d_model, d_model, bias=bias)
        self.k = nn.Linear(d_model, d_model, bias=bias)
        self.v = nn.Linear(d_model, d_model, bias=bias)
        self.o = nn.Linear(d_model, d_model, bias=bias)

    def forward(
        self,
        query,
        key,
        value,
        attn_mask=None,
        key_padding_mask=None,
        need_weights=True,
        average_attn_weights=True,
    ):
        seq_len, batch, _ = query.shape
        d = self.d_model

        Q = self.q(query)
        K = self.k(key)
        V = self.v(value)

        Q = Q.view(seq_len, batch, self.n_head, self.d_k).permute(1, 2, 0, 3)
        K = K.view(K.size(0), batch, self.n_head, self.d_k).permute(1, 2, 0, 3)
        V = V.view(V.size(0), batch, self.n_head, self.d_k).permute(1, 2, 0, 3)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        if attn_mask is not None:
            scores = scores + attn_mask
        if key_padding_mask is not None:
            scores = scores.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2), float("-inf")
            )

        attn_weights = F.softmax(scores, dim=-1)
        context = torch.matmul(attn_weights, V)
        context = context.permute(2, 0, 1, 3).contiguous().view(seq_len, batch, d)
        output = self.o(context)

        if not need_weights:
            return output, None
        if average_attn_weights:
            return output, attn_weights.mean(dim=1)
        return output, attn_weights


# ---------------------------------------------------------------------------
# Weight copy helpers (module-level)
# ---------------------------------------------------------------------------


def copy_weights_from_pytorch_mha(pt_mha, custom_mha):
    d = custom_mha.d_model
    with torch.no_grad():
        custom_mha.q.weight.copy_(pt_mha.in_proj_weight[:d])
        custom_mha.k.weight.copy_(pt_mha.in_proj_weight[d : 2 * d])
        custom_mha.v.weight.copy_(pt_mha.in_proj_weight[2 * d :])
        custom_mha.o.weight.copy_(pt_mha.out_proj.weight)
        if pt_mha.in_proj_bias is not None:
            custom_mha.q.bias.copy_(pt_mha.in_proj_bias[:d])
            custom_mha.k.bias.copy_(pt_mha.in_proj_bias[d : 2 * d])
            custom_mha.v.bias.copy_(pt_mha.in_proj_bias[2 * d :])
            custom_mha.o.bias.copy_(pt_mha.out_proj.bias)


def copy_weights_to_pytorch_mha(custom_mha, pt_mha):
    d = custom_mha.d_model
    with torch.no_grad():
        pt_mha.in_proj_weight[:d].copy_(custom_mha.q.weight)
        pt_mha.in_proj_weight[d : 2 * d].copy_(custom_mha.k.weight)
        pt_mha.in_proj_weight[2 * d :].copy_(custom_mha.v.weight)
        pt_mha.out_proj.weight.copy_(custom_mha.o.weight)
        if pt_mha.in_proj_bias is not None:
            pt_mha.in_proj_bias[:d].copy_(custom_mha.q.bias)
            pt_mha.in_proj_bias[d : 2 * d].copy_(custom_mha.k.bias)
            pt_mha.in_proj_bias[2 * d :].copy_(custom_mha.v.bias)
            pt_mha.out_proj.bias.copy_(custom_mha.o.bias)


# ---------------------------------------------------------------------------
# State-dict conversion helpers
# ---------------------------------------------------------------------------


def copy_from_pytorch_state_dict(state_dict):
    """Convert a PyTorch MHA state_dict to our custom MHA format.

    Maps:
        *.in_proj_weight  -> *.q.weight, *.k.weight, *.v.weight
        *.out_proj.weight -> *.o.weight
        *.in_proj_bias    -> *.q.bias,   *.k.bias,   *.v.bias
        *.out_proj.bias   -> *.o.bias
    Non-MHA keys are passed through unchanged.
    """
    mha_pattern = re.compile(
        r"^(.*)\.(?:in_proj_weight|in_proj_bias|out_proj\.weight|out_proj\.bias)$"
    )

    prefixes = set()
    for key in state_dict:
        m = mha_pattern.match(key)
        if m:
            prefixes.add(m.group(1))

    new_state_dict = {}

    # Pass through non-MHA keys
    for key in state_dict:
        if not mha_pattern.match(key):
            new_state_dict[key] = state_dict[key]

    for prefix in prefixes:
        in_proj_weight = state_dict[f"{prefix}.in_proj_weight"]
        d = in_proj_weight.size(0) // 3

        new_state_dict[f"{prefix}.q.weight"] = in_proj_weight[:d].clone()
        new_state_dict[f"{prefix}.k.weight"] = in_proj_weight[d : 2 * d].clone()
        new_state_dict[f"{prefix}.v.weight"] = in_proj_weight[2 * d :].clone()
        new_state_dict[f"{prefix}.o.weight"] = state_dict[
            f"{prefix}.out_proj.weight"
        ].clone()

        if f"{prefix}.in_proj_bias" in state_dict:
            in_proj_bias = state_dict[f"{prefix}.in_proj_bias"]
            new_state_dict[f"{prefix}.q.bias"] = in_proj_bias[:d].clone()
            new_state_dict[f"{prefix}.k.bias"] = in_proj_bias[d : 2 * d].clone()
            new_state_dict[f"{prefix}.v.bias"] = in_proj_bias[2 * d :].clone()
            new_state_dict[f"{prefix}.o.bias"] = state_dict[
                f"{prefix}.out_proj.bias"
            ].clone()

    return new_state_dict


def copy_to_pytorch_state_dict(state_dict):
    """Convert our custom MHA state_dict back to PyTorch MHA format.

    Maps:
        *.q.weight, *.k.weight, *.v.weight -> *.in_proj_weight
        *.o.weight                          -> *.out_proj.weight
        *.q.bias,   *.k.bias,   *.v.bias   -> *.in_proj_bias
        *.o.bias                            -> *.out_proj.bias
    Non-MHA keys are passed through unchanged.
    """
    mha_pattern = re.compile(
        r"^(.*)\.(?:q\.weight|k\.weight|v\.weight|o\.weight"
        r"|q\.bias|k\.bias|v\.bias|o\.bias)$"
    )

    prefixes = set()
    for key in state_dict:
        m = mha_pattern.match(key)
        if m:
            prefixes.add(m.group(1))

    new_state_dict = {}

    for key in state_dict:
        if not mha_pattern.match(key):
            new_state_dict[key] = state_dict[key]

    for prefix in prefixes:
        new_state_dict[f"{prefix}.in_proj_weight"] = torch.cat(
            [
                state_dict[f"{prefix}.q.weight"],
                state_dict[f"{prefix}.k.weight"],
                state_dict[f"{prefix}.v.weight"],
            ],
            dim=0,
        )
        new_state_dict[f"{prefix}.out_proj.weight"] = state_dict[
            f"{prefix}.o.weight"
        ].clone()

        if f"{prefix}.q.bias" in state_dict:
            new_state_dict[f"{prefix}.in_proj_bias"] = torch.cat(
                [
                    state_dict[f"{prefix}.q.bias"],
                    state_dict[f"{prefix}.k.bias"],
                    state_dict[f"{prefix}.v.bias"],
                ],
                dim=0,
            )
            new_state_dict[f"{prefix}.out_proj.bias"] = state_dict[
                f"{prefix}.o.bias"
            ].clone()

    return new_state_dict


def swap_mha(model):
    """Recursively replace all nn.MultiheadAttention with our MultiHeadAttention."""
    for name, module in model.named_children():
        if isinstance(module, nn.MultiheadAttention):
            custom = MultiHeadAttentionSplit(
                d_model=module.embed_dim,
                n_head=module.num_heads,
                bias=module.in_proj_bias is not None,
            )
            copy_weights_from_pytorch_mha(module, custom)
            setattr(model, name, custom)
        else:
            swap_mha(module)
    return model


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestMultiHeadAttention(unittest.TestCase):
    D_MODEL = 768
    N_HEAD = 4
    ATOL = 1e-6

    def _make_pair(self, bias, direction="from_pytorch"):
        pt = nn.MultiheadAttention(self.D_MODEL, self.N_HEAD, bias=bias)
        custom = MultiHeadAttentionSplit(self.D_MODEL, self.N_HEAD, bias=bias)
        if direction == "from_pytorch":
            copy_weights_from_pytorch_mha(pt, custom)
        else:
            copy_weights_to_pytorch_mha(custom, pt)
        pt.eval()
        custom.eval()
        return pt, custom

    def _assert_output_match(self, pt, custom, q, k, v, **kwargs):
        pt_out, pt_attn = pt(
            q, k, v, need_weights=True, average_attn_weights=True, **kwargs
        )
        custom_out, custom_attn = custom(q, k, v, **kwargs)
        torch.testing.assert_close(custom_out, pt_out, atol=self.ATOL, rtol=0)
        torch.testing.assert_close(custom_attn, pt_attn, atol=self.ATOL, rtol=0)

    # --- from_pytorch direction ---
    def test_self_attention(self):
        for bias in [False, True]:
            with self.subTest(bias=bias):
                pt, custom = self._make_pair(bias)
                x = torch.randn(10, 2, self.D_MODEL)
                self._assert_output_match(pt, custom, x, x, x)

    def test_cross_attention(self):
        for bias in [False, True]:
            with self.subTest(bias=bias):
                pt, custom = self._make_pair(bias)
                q = torch.randn(5, 2, self.D_MODEL)
                kv = torch.randn(12, 2, self.D_MODEL)
                self._assert_output_match(pt, custom, q, kv, kv)

    def test_causal_mask(self):
        for bias in [False, True]:
            with self.subTest(bias=bias):
                pt, custom = self._make_pair(bias)
                x = torch.randn(10, 2, self.D_MODEL)
                mask = nn.Transformer.generate_square_subsequent_mask(10)
                self._assert_output_match(pt, custom, x, x, x, attn_mask=mask)

    def test_key_padding_mask(self):
        for bias in [False, True]:
            with self.subTest(bias=bias):
                pt, custom = self._make_pair(bias)
                x = torch.randn(10, 2, self.D_MODEL)
                kpm = torch.zeros(2, 10, dtype=torch.bool)
                kpm[0, 7:] = True
                kpm[1, 9:] = True
                self._assert_output_match(pt, custom, x, x, x, key_padding_mask=kpm)

    # --- to_pytorch direction ---
    def test_to_pytorch_self_attention(self):
        for bias in [False, True]:
            with self.subTest(bias=bias):
                pt, custom = self._make_pair(bias, direction="to_pytorch")
                x = torch.randn(10, 2, self.D_MODEL)
                self._assert_output_match(pt, custom, x, x, x)

    def test_to_pytorch_cross_attention(self):
        for bias in [False, True]:
            with self.subTest(bias=bias):
                pt, custom = self._make_pair(bias, direction="to_pytorch")
                q = torch.randn(5, 2, self.D_MODEL)
                kv = torch.randn(12, 2, self.D_MODEL)
                self._assert_output_match(pt, custom, q, kv, kv)

    # --- roundtrip ---
    def test_roundtrip_weights(self):
        for bias in [False, True]:
            with self.subTest(bias=bias):
                pt1 = nn.MultiheadAttention(self.D_MODEL, self.N_HEAD, bias=bias)
                custom = MultiHeadAttentionSplit(self.D_MODEL, self.N_HEAD, bias=bias)
                pt2 = nn.MultiheadAttention(self.D_MODEL, self.N_HEAD, bias=bias)
                copy_weights_from_pytorch_mha(pt1, custom)
                copy_weights_to_pytorch_mha(custom, pt2)
                torch.testing.assert_close(pt1.in_proj_weight, pt2.in_proj_weight)
                torch.testing.assert_close(pt1.out_proj.weight, pt2.out_proj.weight)
                if bias:
                    torch.testing.assert_close(pt1.in_proj_bias, pt2.in_proj_bias)
                    torch.testing.assert_close(pt1.out_proj.bias, pt2.out_proj.bias)

    def test_param_count(self):
        for bias in [False, True]:
            with self.subTest(bias=bias):
                pt, custom = self._make_pair(bias)
                pt_p = sum(p.numel() for p in pt.parameters())
                cu_p = sum(p.numel() for p in custom.parameters())
                self.assertEqual(pt_p, cu_p)

    def test_state_dict_roundtrip(self):
        for bias in [False, True]:
            with self.subTest(bias=bias):
                model = nn.Transformer(
                    d_model=self.D_MODEL,
                    nhead=self.N_HEAD,
                    num_encoder_layers=2,
                    num_decoder_layers=2,
                    bias=bias,
                )
                sd = model.state_dict()
                converted = copy_from_pytorch_state_dict(sd)
                restored = copy_to_pytorch_state_dict(converted)
                for key in sd:
                    torch.testing.assert_close(
                        sd[key], restored[key], msg=f"Mismatch on {key}"
                    )


if __name__ == "__main__":
    torch.manual_seed(42)
    unittest.main(verbosity=2)
