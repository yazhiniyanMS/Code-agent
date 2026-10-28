#!/bin/bash
# V11: publish a verified checkpoint to GitHub with Git LFS, then download it back and check every hash.
#
# BLOCKED in the V11 session: the environment's network policy refuses lfs.github.com (the LFS verify
# endpoint), so an upload is never registered; creating releases is refused for the session type. Allow
# lfs.github.com under the environment's Network access, then run this. See V11_REPORT.md section 7.
#
# usage: publish_lfs.sh NAME MODEL_DIR RECORD_JSON "LINEAGE" [--dry-run]
#   NAME        directory name under artifacts/v11/ (e.g. ycode-v11-best-experimental-r1)
#   MODEL_DIR   a restored checkpoint dir (e.g. a .ypk unpacked with `python -m ycode.lm.packckpt unpack`)
#   RECORD_JSON the run's result.json (the evaluation summary goes into MANIFEST.json)
#   --dry-run   everything but the push and the download check
#
# The disk has no room for 3 GB, so the standalone export and the LFS object store live on the RAM disk;
# only the LFS pointer files (~130 bytes each), MANIFEST.json and .gitattributes are committed to Git.
set -euo pipefail
name=$1 model=$2 record=$3 lineage=$4 dry=${5:-}
repo=$(git rev-parse --show-toplevel)
stage=/dev/shm/ycode-publish
lfs=$stage/lfs-objects
branch=$(git -C "$repo" rev-parse --abbrev-ref HEAD)
mkdir -p "$stage" "$lfs"

# 1. a standalone bf16 checkpoint (round 54 folded in), shards under 1.9 GB, MANIFEST.json with every file's
#    sha256 and the tensor content sha256; then loaded back in a separate process and checked
python3 "$repo/experiments/v11/package_checkpoint.py" "$model" "$stage" --name "$name" --record "$record" --lineage "$lineage"
python3 "$repo/experiments/v11/package_checkpoint.py" "$model" "$stage" --name "$name" --verify-only

# 2. LFS pointers into the repository (objects stored on the RAM disk), MANIFEST.json as an ordinary file
dst="$repo/artifacts/v11/$name"
mkdir -p "$dst"
grep -qx "artifacts/v11/\*\*/\*.pt filter=lfs diff=lfs merge=lfs -text" "$repo/.gitattributes" 2>/dev/null \
  || echo "artifacts/v11/**/*.pt filter=lfs diff=lfs merge=lfs -text" >> "$repo/.gitattributes"
for f in "$stage/$name"/*.pt; do
  git -C "$repo" -c lfs.storage="$lfs" lfs clean -- "$f" < "$f" > "$dst/$(basename "$f")"
done
cp "$stage/$name/MANIFEST.json" "$dst/MANIFEST.json"
git -C "$repo" add .gitattributes "artifacts/v11/$name"
git -C "$repo" -c lfs.storage="$lfs" lfs ls-files | grep "artifacts/v11/$name" || { echo "pointers not recognised by git-lfs"; exit 1; }
git -C "$repo" commit -q -m "V11: publish $name through Git LFS (standalone bf16 checkpoint, MANIFEST.json with SHA-256)"
[ "$dry" = "--dry-run" ] && { echo "dry run: committed locally, not pushed"; exit 0; }

# 3. push: the pre-push hook uploads the LFS objects (needs lfs.github.com)
git -C "$repo" -c lfs.storage="$lfs" push -u origin "$branch"

# 4. download every object back from GitHub into a fresh store and compare with the manifest
check=$stage/download-check
rm -rf "$check" && mkdir -p "$check"
git -C "$repo" -c lfs.storage="$check" lfs fetch origin "$branch" --include="artifacts/v11/$name/*"
python3 - "$dst/MANIFEST.json" "$check" "$dst" <<'EOF'
import hashlib, json, sys
from pathlib import Path
manifest, store, dst = json.loads(Path(sys.argv[1]).read_text()), Path(sys.argv[2]), Path(sys.argv[3])
for f in manifest["files"]:
    oid = next(l.split(":", 1)[1] for l in (dst / f["name"]).read_text().splitlines() if l.startswith("oid sha256:"))
    obj = store / "objects" / oid[:2] / oid[2:4] / oid  # lfs.storage keeps objects under objects/
    h = hashlib.sha256()
    with obj.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    assert h.hexdigest() == f["sha256"] == oid, (f["name"], h.hexdigest(), f["sha256"])
    print("downloaded and verified", f["name"], f["sha256"])
EOF
echo "published: https://github.com/$(git -C "$repo" remote get-url origin | sed -E 's#.*github.com[:/]##; s#\.git$##')/tree/$branch/artifacts/v11/$name"
