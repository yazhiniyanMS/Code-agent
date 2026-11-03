"""Find checkpoints by identity instead of by the absolute path recorded when they were written.

Packs and compact checkpoints refer to the checkpoints they need by the path those had on the machine that
wrote them: a delta pack names its parent ``{"path", "content_sha256"}``, and a pack or compact checkpoint
names its base ``{"path", "size"}`` (e.g. /home/user/lmwork/loop/anchor). On another machine those paths do
not exist. With a checkpoint directory configured (``YCODE_CHECKPOINT_DIR``), every such reference is looked
up in that directory by identity instead, at any depth below it:

- a pack by its content hash (the header's ``content_sha256``, a hash of the tensors themselves);
- a base by its model.pt: the file's size and its SHA-256, which must be known: from the spec
  (``anchor_checkpoint``, ``init_checkpoint.base``; the experiment runner passes them on to the processes it
  starts in ``YCODE_BASE_SHA256``) or recorded in the base's own info.json when it was saved.

Nothing is ever used in a checkpoint's place: a reference that is not found, or found with another hash,
raises (``CheckpointNotFound``, ``CheckpointMismatch``) and says what was looked for and where. Unpacking
still checks every tensor against the pack's content hash, so a wrong base or parent cannot load quietly.
Without a checkpoint directory, the recorded paths are used, as before.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path

ENV = "YCODE_CHECKPOINT_DIR"
ENV_BASES = "YCODE_BASE_SHA256"


class CheckpointError(Exception):
    pass


class CheckpointNotFound(CheckpointError, FileNotFoundError):
    pass


class CheckpointMismatch(CheckpointError, ValueError):
    pass


def checkpoint_dir() -> Path | None:
    """The configured checkpoint directory, or None (recorded paths are used)."""
    value = os.environ.get(ENV)
    if not value:
        return None
    root = Path(value)
    if not root.is_dir():
        raise CheckpointNotFound(f"{ENV}={value} is not a directory")
    return root


def pack_header(path: Path | str) -> dict:
    """A pack's header (content hash, parent reference, entries), read from the end of the file alone."""
    with Path(path).open("rb") as f:
        f.seek(-8, os.SEEK_END)
        n = int.from_bytes(f.read(8), "little")
        f.seek(-8 - n, os.SEEK_END)
        return json.loads(f.read(n))


_sha256_cache: dict[tuple, str] = {}


def file_sha256(path: Path) -> str:
    """SHA-256 of a file, remembered per (path, size, mtime): a 5 GB base is hashed once per process."""
    stat = path.stat()
    key = (str(path.resolve()), stat.st_size, stat.st_mtime_ns)
    if key not in _sha256_cache:
        h = hashlib.sha256()
        with path.open("rb") as f:
            for block in iter(lambda: f.read(1 << 24), b""):
                h.update(block)
        _sha256_cache[key] = h.hexdigest()
    return _sha256_cache[key]


def _shown(path: Path, root: Path) -> str:
    return str(path.relative_to(root)) if path.is_relative_to(root) else str(path)


def find_pack(content_sha256: str, root: Path, *, hint: str | None = None, what: str = "checkpoint") -> Path:
    """The pack under ``root`` whose content hash is ``content_sha256``: ``root/hint`` if given and it
    matches, otherwise any .ypk file below ``root`` that does."""
    looked: dict[Path, str] = {}
    candidates = ([root / hint] if hint else []) + sorted(root.rglob("*.ypk"))
    for path in candidates:
        if path in looked or not path.is_file():
            continue
        try:
            found = pack_header(path)["content_sha256"]
        except (OSError, ValueError, KeyError) as exc:
            looked[path] = f"unreadable: {exc}"
            continue
        if found == content_sha256:
            return path
        looked[path] = f"content {found[:12]}..."
    wanted = f"{what} (content {content_sha256[:12]}...)"
    if hint and (root / hint).is_file():
        raise CheckpointMismatch(f"{wanted}: {root / hint} is another checkpoint ({looked[root / hint]}), and no "
                                 f"other pack under {root} has this content")
    seen = "; ".join(f"{_shown(p, root)} ({h})" for p, h in looked.items()) or "no .ypk files"
    raise CheckpointNotFound(f"{wanted} not found under {root} (looked at: {seen})")


def expected_bases() -> set[str]:
    """SHA-256s of the bases checked for this process and its children (``YCODE_BASE_SHA256``, comma-separated;
    the experiment runner sets it after checking the spec's checkpoints)."""
    return {h for h in os.environ.get(ENV_BASES, "").split(",") if h}


def _recorded_sha256(model_dir: Path) -> str | None:
    info = model_dir / "info.json"
    try:
        return json.loads(info.read_text()).get("sha256") if info.is_file() else None
    except (OSError, ValueError):
        return None


def find_base(ref: dict, root: Path, *, expected_sha256: str | None = None, hint: str | None = None,
              what: str = "base checkpoint") -> Path:
    """The checkpoint directory under ``root`` that a base reference ``{"path", "size"}`` means: a model.pt of
    that size whose SHA-256 is checked, against ``expected_sha256``, else ``YCODE_BASE_SHA256``, else the hash
    recorded in its own info.json when it was saved. A base that cannot be checked is not used: a pack's
    content hash covers only the layers it stores, and the frozen ones come from the base."""
    size = ref.get("size")
    expected = {expected_sha256} if expected_sha256 else expected_bases()
    first = [root / hint / "model.pt"] if hint else []
    files = first + sorted(p for p in root.rglob("model.pt") if not first or p != first[0])
    sized = [p for p in files if p.is_file() and (size is None or p.stat().st_size == size)]
    fits = [p for p in sized
            if file_sha256(p) in (expected or {_recorded_sha256(p.parent)} - {None})]
    if fits:  # several only if they are byte-identical copies
        return fits[0].parent
    wanted = (f"{what} (recorded as {ref.get('path')}, model.pt of {size} bytes"
              + (f", sha256 {', '.join(h[:12] + '...' for h in sorted(expected))})" if expected else ")"))
    if sized and not expected:
        raise CheckpointMismatch(f"{wanted}: cannot check " + ", ".join(_shown(p.parent, root) for p in sized)
                                 + f": no SHA-256 is expected (name the base in the spec, or set {ENV_BASES}) and"
                                 " none recorded in its info.json matches")
    if sized:
        raise CheckpointMismatch(f"{wanted}: only other checkpoints under {root} have that size ("
                                 + ", ".join(f"{_shown(p.parent, root)}: sha256 {file_sha256(p)[:12]}..." for p in sized)
                                 + ")")
    raise CheckpointNotFound(f"{wanted} not found under {root}")


def base_dir(ref: dict) -> Path:
    """Where a base reference's checkpoint is: looked up by identity under the checkpoint directory if one is
    configured, else the recorded path."""
    root = checkpoint_dir()
    return find_base(ref, root) if root is not None else Path(ref["path"])


def parent_path(ref: dict) -> Path:
    """Where a delta pack's parent ``{"path", "content_sha256"}`` is (as ``base_dir``)."""
    root = checkpoint_dir()
    if root is None:
        return Path(ref["path"])
    return find_pack(ref["content_sha256"], root, what=f"parent pack (recorded at {ref['path']})")


@dataclass
class Link:
    role: str  # "checkpoint", "parent 1", ..., "base"
    path: Path
    identity: str  # a pack's content hash, a base's model.pt SHA-256 (or its size, if none was expected)

    def record(self) -> dict:
        return {"role": self.role, "path": str(self.path), "identity": self.identity}


def resolve_pack(ref: dict, root: Path | None) -> list[Link]:
    """Locate a pack named by identity in a spec, and everything it needs, before anything is trained on it.

    ``ref``: ``{"name", "content_sha256", "file" (optional hint, relative to the checkpoint directory),
    "parents": [content hashes, nearest first], "base": {"sha256", "size", "file"}}``. Every pack in the
    chain must name the parent the spec expects, and the chain must end where the spec says; every base must
    have the expected SHA-256. These are header checks; loading then checks every tensor."""
    from ycode.lm import packckpt

    name = ref.get("name", "checkpoint")
    if root is None:
        raise CheckpointNotFound(f"{name} is named by its content hash; set {ENV} to the directory that holds it "
                                 "and its parents and base")
    path = find_pack(ref["content_sha256"], root, hint=ref.get("file"), what=name)
    chain = [Link("checkpoint", path, ref["content_sha256"])]
    expected = list(ref.get("parents", []))
    while True:
        parent = pack_header(path).get("parent")
        want = expected.pop(0) if expected else None
        if parent is None:
            if want is not None:
                raise CheckpointMismatch(f"{name}: {_shown(path, root)} has no parent, but the spec expects "
                                         f"one with content {want[:12]}...")
            break
        if want is None or parent["content_sha256"] != want:
            raise CheckpointMismatch(f"{name}: {_shown(path, root)} was packed against a parent with content "
                                     f"{parent['content_sha256'][:12]}..., the spec expects "
                                     + (f"{want[:12]}..." if want else "no parent"))
        path = find_pack(want, root, what=f"parent {len(chain)} of {name} (recorded at {parent['path']})")
        chain.append(Link(f"parent {len(chain)}", path, want))
    base = ref.get("base", {})
    refs = {}
    for link in chain:
        meta = packckpt.describe(link.path)[0]
        if meta.get("base"):
            refs[(meta["base"].get("path"), meta["base"].get("size"))] = meta["base"]
    for recorded in refs.values():
        found = find_base({**recorded, "size": base.get("size", recorded.get("size"))}, root,
                          expected_sha256=base.get("sha256"), hint=base.get("file"), what=f"base of {name}")
        chain.append(Link("base", found, base.get("sha256") or f"model.pt of {recorded.get('size')} bytes"))
    return chain


def resolve_base(ref: dict, root: Path | None) -> Path:
    """A spec's ``anchor_checkpoint`` ``{"name", "sha256", "size", "file"}``: its directory under the
    checkpoint directory, with model.pt's SHA-256 checked."""
    name = ref.get("name", "anchor")
    if root is None:
        raise CheckpointNotFound(f"{name} is named by its SHA-256; set {ENV} to the directory that holds it")
    return find_base({"path": ref.get("file"), "size": ref.get("size")}, root, expected_sha256=ref["sha256"],
                     hint=ref.get("file"), what=name)
