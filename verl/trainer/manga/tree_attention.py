"""Differentiable page-trunk attention; no detached or persistent KV cache.

Layout rows are [visible_trunk_length, physical_start, token_count]. The first
row is the original trajectory. Remaining rows are independent corrective leaves.
The layout is passed through decoder/checkpoint kwargs, including recomputation.
"""
import torch


def validate_layout(layout, sequence_length):
    if not layout or layout[0][0:2] != [0, 0] or layout[0][2] <= 0:
        raise ValueError("Tree must start with one nonempty original trajectory")
    root_length = layout[0][2]
    cursor = root_length
    for visible, start, length in layout[1:]:
        if not 0 < visible <= root_length or start != cursor or length <= 0:
            raise ValueError("Invalid corrective tree: leaves must be contiguous and see only the trunk")
        cursor += length
    if cursor != sequence_length:
        raise ValueError("Tree layout does not cover the complete input")


def restore_branch_positions(position_ids, layout):
    """Restore logical M-RoPE/text positions instead of physical packed positions."""
    validate_layout(layout, position_ids.shape[-1])
    result = position_ids.clone()
    for visible, start, length in layout[1:]:
        # Corrections are text only. The previous original token supplies the
        # final 3-axis M-RoPE position and the separate packed text position.
        previous = result[..., visible - 1:visible]
        steps = torch.arange(1, length + 1, device=result.device, dtype=result.dtype)
        result[..., start:start + length] = previous + steps
    return result


class PageTreeAttention:
    def __init__(self, layouts, lengths, device):
        trunk, leaves, keys = [], [], []
        trunk_lengths, leaf_lengths, key_lengths = [], [], []
        base = 0
        for layout, length in zip(layouts, lengths, strict=True):
            validate_layout(layout, length)
            root_length = layout[0][2]
            trunk.extend(range(base, base + root_length))
            trunk_lengths.append(root_length)
            for visible, start, size in layout[1:]:
                branch = list(range(base + start, base + start + size))
                leaves.extend(branch)
                keys.extend(range(base, base + visible))
                keys.extend(branch)
                leaf_lengths.append(size)
                key_lengths.append(visible + size)
            base += length
        self.trunk = torch.tensor(trunk, device=device, dtype=torch.long)
        self.leaves = torch.tensor(leaves, device=device, dtype=torch.long)
        self.keys = torch.tensor(keys, device=device, dtype=torch.long)
        self.trunk_cu, self.trunk_max = self._cu(trunk_lengths, device)
        self.leaf_cu, self.leaf_max = self._cu(leaf_lengths, device)
        self.key_cu, self.key_max = self._cu(key_lengths, device)
        self.calls = 0

    @staticmethod
    def _cu(lengths, device):
        offsets = [0]
        for length in lengths:
            offsets.append(offsets[-1] + length)
        return torch.tensor(offsets, dtype=torch.int32, device=device), max(lengths, default=0)

    def forward(self, unused_attention_function, query, key, value, *args, **kwargs):
        from flash_attn import flash_attn_varlen_func
        if args or kwargs.get("dropout", 0.0) or kwargs.get("sliding_window") or kwargs.get("softcap"):
            raise ValueError("Page tree requires zero attention dropout, full attention and no softcap")
        if query.shape[0] != 1:
            raise ValueError("Page trees must be packed into one token dimension")
        self.calls += 1
        q, k, v = (x.squeeze(0).transpose(0, 1).contiguous() for x in (query, key, value))
        scale = kwargs.get("scaling", q.shape[-1] ** -0.5)
        root_out = flash_attn_varlen_func(
            q.index_select(0, self.trunk), k.index_select(0, self.trunk), v.index_select(0, self.trunk),
            self.trunk_cu, self.trunk_cu, self.trunk_max, self.trunk_max,
            dropout_p=0.0, softmax_scale=scale, causal=True)
        output = q.new_zeros(q.shape).index_copy(0, self.trunk, root_out)
        if self.leaf_max:
            # FlashAttention >=2.1 bottom-right causal alignment: each leaf query
            # sees its original prefix and its own preceding tokens, never peers
            # or the original future. index_select backward sums into shared K/V.
            leaf_out = flash_attn_varlen_func(
                q.index_select(0, self.leaves), k.index_select(0, self.keys), v.index_select(0, self.keys),
                self.leaf_cu, self.key_cu, self.leaf_max, self.key_max,
                dropout_p=0.0, softmax_scale=scale, causal=True)
            output = output.index_copy(0, self.leaves, leaf_out)
        return output.unsqueeze(0)
