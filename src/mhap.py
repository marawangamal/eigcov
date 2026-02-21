import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import re
import unittest


class MultiHeadAttentionPacked(nn.Module):
    def __init__(self, d_model, n_head, bias=False):
        super().__init__()
        assert d_model % n_head == 0
        self.d_model = d_model
        self.n_head = n_head
        self.d_k = d_model // n_head
        self.qk = nn.Linear(d_model, 2 * d_model, bias=bias)
        self.vot = nn.Linear(d_model, 2 * d_model, bias=bias)

    def forward(self, query, key, value, attn_mask=None, key_padding_mask=None):
        seq_len, batch, _ = query.shape
        d = self.d_model
        # [Do,Di]
        Wq = self.qk.weight[:d]
        Wk = self.qk.weight[d:]
        Wv = self.vot.weight[:d]
        Wo = self.vot.weight[d:].T
        bq = bk = bv = bo = None
        if self.qk.bias is not None:
            bq, bk = self.qk.bias[:d], self.qk.bias[d:]
            bv, bo = self.vot.bias[:d], self.vot.bias[d:]
        Q = F.linear(query, Wq, bq)
        K = F.linear(key, Wk, bk)
        V = F.linear(value, Wv, bv)
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
        output = F.linear(context, Wo, bo)
        attn_weights_avg = attn_weights.mean(dim=1)
        return output, attn_weights_avg


def copy_weights_from_pytorch_mha(pt_mha, custom_mha):
    d = custom_mha.d_model
    with torch.no_grad():
        custom_mha.qk.weight[:d].copy_(pt_mha.in_proj_weight[:d])
        custom_mha.qk.weight[d:].copy_(pt_mha.in_proj_weight[d : 2 * d])
        custom_mha.vot.weight[:d].copy_(pt_mha.in_proj_weight[2 * d :])
        custom_mha.vot.weight[d:].T.copy_(pt_mha.out_proj.weight)
        if pt_mha.in_proj_bias is not None:
            custom_mha.qk.bias[:d].copy_(pt_mha.in_proj_bias[:d])
            custom_mha.qk.bias[d:].copy_(pt_mha.in_proj_bias[d : 2 * d])
            custom_mha.vot.bias[:d].copy_(pt_mha.in_proj_bias[2 * d :])
            custom_mha.vot.bias[d:].copy_(pt_mha.out_proj.bias)


def copy_weights_to_pytorch_mha(custom_mha, pt_mha):
    d = custom_mha.d_model
    with torch.no_grad():
        pt_mha.in_proj_weight[:d].copy_(custom_mha.qk.weight[:d])
        pt_mha.in_proj_weight[d : 2 * d].copy_(custom_mha.qk.weight[d:])
        pt_mha.in_proj_weight[2 * d :].copy_(custom_mha.vot.weight[:d])
        pt_mha.out_proj.weight.copy_(custom_mha.vot.weight[d:].T)
        if pt_mha.in_proj_bias is not None:
            pt_mha.in_proj_bias[:d].copy_(custom_mha.qk.bias[:d])
            pt_mha.in_proj_bias[d : 2 * d].copy_(custom_mha.qk.bias[d:])
            pt_mha.in_proj_bias[2 * d :].copy_(custom_mha.vot.bias[:d])
            pt_mha.out_proj.bias.copy_(custom_mha.vot.bias[d:])


def copy_from_pytorch_state_dict(state_dict):
    """Convert a PyTorch MHA state_dict to our custom MHA format.

    Finds all keys matching *.in_proj_weight, *.out_proj.weight etc.
    and remaps them to *.qk.weight, *.vot.weight etc.
    Non-MHA keys are passed through unchanged.
    """
    # Match any prefix ending with in_proj_weight or out_proj.weight/bias
    mha_pattern = re.compile(
        r"^(.*)\.(?:in_proj_weight|in_proj_bias|out_proj\.weight|out_proj\.bias)$"
    )

    # Group MHA keys by prefix
    prefixes = set()
    for key in state_dict:
        m = mha_pattern.match(key)
        if m:
            prefixes.add(m.group(1))

    new_state_dict = {}

    # Copy non-MHA keys unchanged
    for key in state_dict:
        if not mha_pattern.match(key):
            new_state_dict[key] = state_dict[key]

    # Convert each MHA
    for prefix in prefixes:
        in_proj_weight = state_dict[f"{prefix}.in_proj_weight"]
        d = in_proj_weight.size(0) // 3

        new_state_dict[f"{prefix}.qk.weight"] = torch.cat(
            [
                in_proj_weight[:d],
                in_proj_weight[d : 2 * d],
            ],
            dim=0,
        )

        out_proj_weight = state_dict[f"{prefix}.out_proj.weight"]
        new_state_dict[f"{prefix}.vot.weight"] = torch.cat(
            [
                in_proj_weight[2 * d :],
                out_proj_weight.T,
            ],
            dim=0,
        )

        if f"{prefix}.in_proj_bias" in state_dict:
            in_proj_bias = state_dict[f"{prefix}.in_proj_bias"]
            new_state_dict[f"{prefix}.qk.bias"] = torch.cat(
                [
                    in_proj_bias[:d],
                    in_proj_bias[d : 2 * d],
                ],
                dim=0,
            )

            new_state_dict[f"{prefix}.vot.bias"] = torch.cat(
                [
                    in_proj_bias[2 * d :],
                    state_dict[f"{prefix}.out_proj.bias"],
                ],
                dim=0,
            )

    return new_state_dict


def copy_to_pytorch_state_dict(state_dict):
    """Convert our custom MHA state_dict back to PyTorch MHA format.

    Finds all keys matching *.qk.weight, *.vot.weight etc.
    and remaps them to *.in_proj_weight, *.out_proj.weight etc.
    Non-MHA keys are passed through unchanged.
    """
    mha_pattern = re.compile(r"^(.*)\.(?:qk\.weight|qk\.bias|vot\.weight|vot\.bias)$")

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
        qk_weight = state_dict[f"{prefix}.qk.weight"]
        vot_weight = state_dict[f"{prefix}.vot.weight"]
        d = qk_weight.size(0) // 2

        new_state_dict[f"{prefix}.in_proj_weight"] = torch.cat(
            [
                qk_weight[:d],
                qk_weight[d:],
                vot_weight[:d],
            ],
            dim=0,
        )

        new_state_dict[f"{prefix}.out_proj.weight"] = vot_weight[d:].T

        if f"{prefix}.qk.bias" in state_dict:
            qk_bias = state_dict[f"{prefix}.qk.bias"]
            vot_bias = state_dict[f"{prefix}.vot.bias"]

            new_state_dict[f"{prefix}.in_proj_bias"] = torch.cat(
                [
                    qk_bias[:d],
                    qk_bias[d:],
                    vot_bias[:d],
                ],
                dim=0,
            )

            new_state_dict[f"{prefix}.out_proj.bias"] = vot_bias[d:]

    return new_state_dict


def swap_mha(model):
    """Recursively replace all nn.MultiheadAttention with our MultiHeadAttention."""
    for name, module in model.named_children():
        if isinstance(module, nn.MultiheadAttention):
            custom = MultiHeadAttentionPacked(
                d_model=module.embed_dim,
                n_head=module.num_heads,
                bias=module.in_proj_bias is not None,
            )
            copy_weights_from_pytorch_mha(module, custom)
            setattr(model, name, custom)
        else:
            swap_mha(module)
    return model


class TestMultiHeadAttention(unittest.TestCase):
    D_MODEL = 768
    N_HEAD = 4
    ATOL = 1e-6

    def _make_pair(self, bias, direction="from_pytorch"):
        """Create matched PyTorch and custom MHA with shared weights."""
        pt = nn.MultiheadAttention(self.D_MODEL, self.N_HEAD, bias=bias)
        custom = MultiHeadAttentionPacked(self.D_MODEL, self.N_HEAD, bias=bias)
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
                custom = MultiHeadAttentionPacked(self.D_MODEL, self.N_HEAD, bias=bias)
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
