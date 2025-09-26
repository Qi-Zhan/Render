import argparse
import itertools

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention import flex_attention


parser = argparse.ArgumentParser(description='Minimum MHA / SDPA / Flex Attention mask test')
parser.add_argument('--test-flex-attention', action='store_true',
                    help='assert_close for the flex attention output to the rest.')
parser.add_argument('--mask', action='store_true',
                    help='Test with a simple attention mask.')
parser.add_argument('--compile', action='store_true',
                    help='Compile the model.')
parser.add_argument('--high-precision', action='store_true',
                    help='Set float32 matmul precision to "high" (instead of "highest").')


def apply_attn_in(input, in_w, in_b, num_heads, head_dim):
    b, s, d = input.shape
    qkv = F.linear(input, in_w, in_b)
    qkv = qkv.reshape((b, s, 3, num_heads, head_dim))
    # (b, s, 3, nh, hd) -> (3, b, nh, s, hd)
    return qkv.permute(2, 0, 3, 1, 4)


def apply_attn_out(out, out_w, out_b):
    # (b, nh, s, hd) -> (b, s, nh, hd)
    out = out.transpose(1, 2)
    b, s, nh, hd = out.shape
    out = out.reshape((b, s, nh * hd))
    return F.linear(out, out_w, out_b)


def apply_attention(input, in_w, in_b, out_w, out_b, num_heads, head_dim, attn_mask, block_mask):
    q, k, v = apply_attn_in(input, in_w, in_b, num_heads, head_dim)
    out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
    sdpa_out = apply_attn_out(out, out_w, out_b)
    out = flex_attention.flex_attention(q, k, v, block_mask=block_mask)
    flex_out = apply_attn_out(out, out_w, out_b)
    return sdpa_out, flex_out


class ThreeAttention(nn.Module):

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        attn_mask,
        block_mask,
        test_flex_attention,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.mha = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.attn_mask = attn_mask
        self.block_mask = block_mask
        self.test_flex_attention = test_flex_attention

    def forward(self, x):
        sdpa_out, flex_out = apply_attention(
            x, self.mha.in_proj_weight,
            self.mha.in_proj_bias,
            self.mha.out_proj.weight,
            self.mha.out_proj.bias,
            self.num_heads,
            self.hidden_dim // self.num_heads,
            self.attn_mask, self.block_mask)
        if self.test_flex_attention:
            torch.testing.assert_close(sdpa_out, flex_out)
        mha_out, _ = self.mha(x, x, x, need_weights=False, attn_mask=None if self.attn_mask is None else ~self.attn_mask)
        torch.testing.assert_close(sdpa_out, mha_out)
        return mha_out


def main():
    args = parser.parse_args()
    for args.test_flex_attention, args.mask, args.compile, args.high_precision in itertools.product((False, True), repeat=4):
        try:
            if args.high_precision:
                torch.set_float32_matmul_precision('high')
            else:
                torch.set_float32_matmul_precision('highest')
            hidden_dim = 16
            num_heads = 1
            seq_length = 2
            attn_mask = block_mask = None
            if args.mask:
                attn_mask = torch.eye(seq_length, dtype=torch.bool).cuda()
                def mask_mod(b, h, q_idx, kv_idx):
                    return attn_mask[q_idx][kv_idx]
                block_mask = flex_attention.create_block_mask(
                    mask_mod, B=None, H=None, Q_LEN=seq_length, KV_LEN=seq_length, BLOCK_SIZE=seq_length)
            attn = ThreeAttention(hidden_dim, num_heads, attn_mask, block_mask, args.test_flex_attention).cuda()
            if args.compile:
                attn = torch.compile(attn)
            batch_size = 1
            x = torch.randn(batch_size, seq_length, hidden_dim).cuda()
            attn(x)
        except Exception as ex:
            print(f"{args.test_flex_attention=}, {args.mask=}, {args.compile=}, {args.high_precision=} FAILED!")
            print(ex)


if __name__ == "__main__":
    main()