"""Lossless packing of bf16 fine-tune checkpoints, for a disk that cannot hold them as they are (V10).

A compact bf16 checkpoint (the trained tensors of a fine-tune, referencing its base for the rest) is
stored as the XOR of each tensor's bits with the base's tensor rounded to bf16, split into its high
and low bytes, each zlib-compressed. Fine-tuning moves most weights by less than their bf16 step,
so the XOR is mostly zeros: about 58% of the original size. ``unpack`` restores every tensor bit for
bit; ``content_sha256`` hashes the tensors themselves (names, dtypes, shapes, bytes), so it is the
same for a checkpoint and its unpacked copy, whatever their file bytes.

A checkpoint trained on from another one (S1 from R1) can instead be packed against that parent (a pack or
a checkpoint dir): each tensor is stored as the difference of its bits from the parent's, zigzag-encoded,
with the values grouped by the parent's exponent before compression (a weight's bf16 step, and so how many
steps it moved, depends on its exponent; the parent gives the grouping back when unpacking). For V12-S1, 7
steps after R1, that is 0.42 GB instead of 1.32 GB. Unpacking needs the parent and checks its content hash.

usage: python -m ycode.lm.packckpt pack SRC_DIR DST.ypk [PARENT] | unpack DST.ypk OUT_DIR | verify SRC_DIR DST.ypk
"""
from __future__ import annotations

import hashlib
import io
import json
import shutil
import sys
import zlib
from pathlib import Path

import numpy as np
import torch

MAGIC = b"YPK1"
MAGIC_DELTA = b"YPK2"  # a pack with delta entries: a reader that predates them refuses it outright
LEVEL = 6


def content_sha256(state: dict) -> str:
    h = hashlib.sha256()
    for k in sorted(state):
        t = state[k].contiguous()
        h.update(f"{k}|{t.dtype}|{tuple(t.shape)}|".encode())
        h.update(t.view(torch.uint8).numpy().tobytes() if t.numel() else b"")
    return h.hexdigest()


def _base_state(payload: dict) -> dict:
    path = Path(payload["base"]["path"]) / "model.pt"
    return torch.load(path, map_location="cpu", weights_only=True, mmap=True)["model"]


def _parent_state(parent: Path | str) -> tuple[dict, str]:
    """The tensors of a parent checkpoint (a pack, a checkpoint dir or a model.pt) and their content hash."""
    parent = Path(parent)
    if parent.suffix == ".ypk":
        state = unpacked_state(parent)[0]
    else:
        path = parent / "model.pt" if parent.is_dir() else parent
        state = torch.load(path, map_location="cpu", weights_only=True, mmap=True)["model"]
    return state, content_sha256(state)


def _exponent_order(parent_bits: np.ndarray) -> np.ndarray:
    return np.argsort((parent_bits.view(np.uint16) >> 7) & 0xFF, kind="stable")


def _delta_parts(t: torch.Tensor, parent: torch.Tensor) -> list[bytes]:
    p = parent.contiguous().view(torch.int16).numpy().reshape(-1)
    d = (t.view(torch.int16).numpy().reshape(-1).view(np.uint16) - p.view(np.uint16)).view(np.int16)  # mod 2**16
    z = ((d << 1) ^ (d >> 15)).view(np.uint16)  # zigzag: a move of k steps either way becomes 2k or 2k - 1
    planes = z[_exponent_order(p)].view(np.uint8).reshape(-1, 2)
    return [zlib.compress(planes[:, 0].tobytes(), LEVEL), zlib.compress(planes[:, 1].tobytes(), LEVEL)]


def _restore_delta(parts: list, parent: torch.Tensor, shape: list) -> torch.Tensor:
    p = parent.contiguous().view(torch.int16).numpy().reshape(-1)
    grouped = np.empty((len(parts[0]), 2), dtype=np.uint8)
    grouped[:, 0] = np.frombuffer(parts[0], dtype=np.uint8)
    grouped[:, 1] = np.frombuffer(parts[1], dtype=np.uint8)
    z = np.empty(len(p), dtype=np.uint16)
    z[_exponent_order(p)] = grouped.reshape(-1).view(np.uint16)
    d = (z >> 1) ^ (np.uint16(0) - (z & 1))
    bits = p.view(np.uint16) + d
    return torch.from_numpy(bits.view(np.int16).copy()).view(torch.bfloat16).reshape(shape)


def pack(src: Path | str, dst: Path | str, parent: Path | str | None = None) -> dict:
    """Pack ``src`` (a checkpoint dir or model.pt) into ``dst``; returns the header written. With ``parent``
    (the checkpoint ``src`` was trained from), every bf16 tensor the parent also has is stored as a delta
    from it; the others as usual, against the base."""
    src = Path(src)
    path = src / "model.pt" if src.is_dir() else src
    payload = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    if not payload.get("base"):
        raise ValueError(f"{path} is not base-referenced: nothing to pack against")
    base = _base_state(payload)
    state = payload["model"]
    parent_state, parent_hash = _parent_state(parent) if parent is not None else ({}, None)
    entries = []
    dst = Path(dst)
    tmp = dst.with_suffix(".tmp")
    f = tmp.open("wb")
    f.write(MAGIC if parent is None else MAGIC_DELTA)
    for k, t in state.items():
        t = t.contiguous()
        p = parent_state.get(k)
        if t.dtype == torch.bfloat16 and p is not None and p.dtype == torch.bfloat16 and p.shape == t.shape:
            parts, kind = _delta_parts(t, p), "delta_bf16"
        elif t.dtype == torch.bfloat16 and k in base and base[k].shape == t.shape:
            bits = t.view(torch.int16).numpy() ^ base[k].to(torch.bfloat16).contiguous().view(torch.int16).numpy()
            planes = bits.view(np.uint8).reshape(-1, 2)
            parts = [zlib.compress(planes[:, 0].tobytes(), LEVEL), zlib.compress(planes[:, 1].tobytes(), LEVEL)]
            kind = "xor_bf16"
        else:
            buf = io.BytesIO()
            torch.save(t.clone(), buf)
            parts, kind = [zlib.compress(buf.getvalue(), LEVEL)], "torch"
        entries.append({"key": k, "kind": kind, "shape": list(t.shape), "sizes": [len(p) for p in parts]})
        for part in parts:
            f.write(part)
    meta = {k: v for k, v in payload.items() if k != "model"}
    meta_buf = io.BytesIO()
    torch.save(meta, meta_buf)
    header = {"entries": entries, "meta_size": len(meta_buf.getvalue()), "content_sha256": content_sha256(state),
              "source": str(path), "source_bytes": path.stat().st_size}
    if parent is not None:
        header["parent"] = {"path": str(Path(parent).resolve()), "content_sha256": parent_hash}
    head = json.dumps(header).encode()
    f.write(meta_buf.getvalue() + head + len(head).to_bytes(8, "little"))
    f.close()
    tmp.replace(dst)
    return header


def _read(dst: Path):
    import mmap

    f = Path(dst).open("rb")
    data = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
    if data[:4] not in (MAGIC, MAGIC_DELTA):
        raise ValueError(f"{dst} is not a packed checkpoint")
    n = int.from_bytes(data[-8:], "little")
    header = json.loads(data[-8 - n:-8])
    start = len(data) - 8 - n - header["meta_size"]
    meta = torch.load(io.BytesIO(data[start:start + header["meta_size"]]), map_location="cpu", weights_only=True)
    return data, header, meta, 4


def tensors(dst: Path | str, parent: Path | str | None = None):
    """(meta, header, iterator of (key, tensor)): every tensor restored bit for bit, one at a time. A pack made
    against a parent needs it: ``parent``, or the path recorded when packing; its content hash must match."""
    data, header, meta, pos = _read(Path(dst))
    base = _base_state(meta)
    parent_state = {}
    if "parent" in header:
        parent_state, parent_hash = _parent_state(parent or header["parent"]["path"])
        if parent_hash != header["parent"]["content_sha256"]:
            raise ValueError(f"{dst}: the parent checkpoint differs from the one it was packed against")

    def it(pos=pos):
        for e in header["entries"]:
            parts = []
            for size in e["sizes"]:
                parts.append(zlib.decompress(data[pos:pos + size]))
                pos += size
            yield e["key"], (_restore_delta(parts, parent_state[e["key"]], e["shape"]) if e["kind"] == "delta_bf16"
                             else _restore(e, parts, base))
    return meta, header, it()


def _restore(e: dict, parts: list, base: dict) -> torch.Tensor:
    if e["kind"] == "xor_bf16":
        planes = np.empty((len(parts[0]), 2), dtype=np.uint8)
        planes[:, 0] = np.frombuffer(parts[0], dtype=np.uint8)
        planes[:, 1] = np.frombuffer(parts[1], dtype=np.uint8)
        ref =base[e["key"]].to(torch.bfloat16).contiguous().view(torch.int16).numpy().reshape(-1)
        bits = planes.reshape(-1).view(np.int16) ^ ref
        return torch.from_numpy(bits.copy()).view(torch.bfloat16).reshape(e["shape"])
    if e["kind"] != "torch":
        raise ValueError(f"unknown packed tensor kind {e['kind']!r} for {e['key']}")
    return torch.load(io.BytesIO(parts[0]), map_location="cpu", weights_only=True)


def unpacked_state(dst: Path | str, parent: Path | str | None = None) -> tuple[dict, dict, dict]:
    """(state, meta, header) of a packed checkpoint, every tensor restored bit for bit."""
    meta, header, it = tensors(dst, parent)
    state = dict(it)
    if content_sha256(state) != header["content_sha256"]:
        raise ValueError(f"{dst}: unpacked tensors differ from the packed checkpoint")
    return state, meta, header


def unpack(dst: Path | str, out_dir: Path | str) -> Path:
    """Write the packed checkpoint back as out_dir/model.pt (with info.json if the source had one)."""
    state, meta, header = unpacked_state(dst)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp = out_dir / "model.pt.tmp"
    torch.save({**meta, "model": state}, tmp)
    tmp.replace(out_dir / "model.pt")
    info = Path(header["source"]).parent / "info.json"
    sidecar = Path(dst).with_suffix(".info.json")
    if sidecar.is_file():
        shutil.copy(sidecar, out_dir / "info.json")
    elif info.is_file():
        shutil.copy(info, out_dir / "info.json")
    return out_dir / "model.pt"


def verify(src: Path | str, dst: Path | str) -> str:
    """Raise unless ``dst`` restores ``src``'s tensors and metadata exactly; return the content hash."""
    src = Path(src)
    path = src / "model.pt" if src.is_dir() else src
    payload = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    meta, header, it = tensors(dst)
    keys = []
    for k, t in it:  # one tensor at a time: bit for bit against the source
        s = payload["model"][k]
        same = s.dtype == t.dtype and s.shape == t.shape and torch.equal(
            s.contiguous().view(torch.uint8), t.contiguous().view(torch.uint8))
        if not same:
            raise ValueError(f"tensor {k!r} differs")
        keys.append(k)
    if sorted(keys) != sorted(payload["model"]):
        raise ValueError("tensor names differ")
    if content_sha256(payload["model"]) != header["content_sha256"]:
        raise ValueError("the source changed since it was packed")
    for k, v in payload.items():
        if k != "model" and repr(meta.get(k)) != repr(v):
            raise ValueError(f"metadata {k!r} differs")
    return header["content_sha256"]


def main(argv: list[str]) -> None:
    cmd, *rest = argv
    if cmd == "pack":
        src, dst, *parent = map(Path, rest)
        header = pack(src, dst, *parent)
        info = (src if src.is_dir() else src.parent) / "info.json"
        if info.is_file():
            shutil.copy(info, Path(dst).with_suffix(".info.json"))
        print(json.dumps({"content_sha256": header["content_sha256"], "packed_bytes": Path(dst).stat().st_size,
                          "source_bytes": header["source_bytes"]}))
    elif cmd == "unpack":
        print(unpack(*rest))
    elif cmd == "verify":
        print(verify(*rest))
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main(sys.argv[1:])
