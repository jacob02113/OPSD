"""Opt-in, process-local diagnostic artifacts; no training changes by default."""
import hashlib
import json
import os
from pathlib import Path
from uuid import uuid4


def enabled():
    return bool(os.environ.get("MANGA_DIAG_DIR"))


def frozen():
    return enabled() and os.environ.get("MANGA_DIAG_MODE") == "frozen"


def sequence_key(ids):
    return hashlib.sha256(json.dumps(list(ids)).encode()).hexdigest()[:24]


def write_record(kind, payload):
    if not enabled():
        return None
    root = Path(os.environ["MANGA_DIAG_DIR"]) / kind
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"{os.getpid()}-{uuid4().hex}.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, default=str), encoding="utf-8")
    return path




def record_opsd_actor(data, distill):
    if not enabled():
        return
    responses = data["responses"]
    mask = data["response_mask"]
    if responses.is_nested:
        responses = responses.to_padded_tensor(0)
    if mask.is_nested:
        mask = mask.to_padded_tensor(False)
    for row in range(distill.shape[0]):
        positions = mask[row].nonzero(as_tuple=True)[0]
        n = int(positions[-1].item()) + 1 if positions.numel() else 0
        ids = responses[row, :n].detach().cpu().tolist()
        write_record("actor", dict(sequence_key=sequence_key(ids), response_ids=ids,
                     mode=os.environ.get("MANGA_DIAG_MODE"), objective="opsd",
                     policy_logprobs_available=False,
                     values={"mask": mask[row, :n].detach().cpu().tolist(),
                             "kl": distill[row, :n].detach().float().cpu().tolist()}))
