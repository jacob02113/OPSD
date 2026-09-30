"""Causal boundary labels are the actual next tokens, independent of the teacher."""
import torch
import torch.nn.functional as F


def boundary_cross_entropy(logits, input_ids, causal_mask, student_positions=None):
    flat = logits.squeeze(0)
    positions = causal_mask.nonzero(as_tuple=True)[0]
    labels = input_ids[positions + 1]
    projected = positions
    if student_positions is not None:
        projected = torch.searchsorted(student_positions, positions)
        if (bool((projected >= student_positions.numel()).any()) or
                not torch.equal(student_positions[projected], positions)):
            raise ValueError("Boundary positions missing from selected projections")
    selected = flat.index_select(0, projected).float()
    # Empty selections must retain a gradient path for synthetic DP padding.
    ce = F.cross_entropy(selected, labels, reduction="none")
    losses = flat.new_zeros(input_ids.numel(), dtype=torch.float32).scatter(0, positions, ce)
    correct = flat.new_zeros(input_ids.numel(), dtype=torch.float32).scatter(
        0, positions, (selected.argmax(-1) == labels).float())
    return losses.unsqueeze(0), correct.unsqueeze(0)
