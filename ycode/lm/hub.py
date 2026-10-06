"""Hugging Face Hub storage for YCode-LM: publish models and sync training checkpoints.

``HFHub`` talks to huggingface.co with ``huggingface_hub`` (``pip install huggingface_hub``)
and a write token from ``HF_TOKEN``. ``LocalHub`` implements the same interface on a local
directory; tests use it, and it is handy for dry runs.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path


class Hub:
    """Minimal storage interface: folders under a path inside one repository."""

    repo_id: str

    def upload_folder(self, local_dir: Path, path_in_repo: str, message: str) -> None:
        raise NotImplementedError

    def download_folder(self, path_in_repo: str, local_dir: Path) -> bool:
        """Copy ``path_in_repo`` into ``local_dir``. Returns False if it does not exist."""
        raise NotImplementedError

    def url(self) -> str:
        raise NotImplementedError


class LocalHub(Hub):
    def __init__(self, root: Path, repo_id: str = "local/ycode-lm") -> None:
        self.root = Path(root) / repo_id
        self.repo_id = repo_id
        self.root.mkdir(parents=True, exist_ok=True)

    def upload_folder(self, local_dir: Path, path_in_repo: str, message: str) -> None:
        target = self.root / path_in_repo if path_in_repo else self.root
        if path_in_repo and target.exists():
            shutil.rmtree(target)
        shutil.copytree(local_dir, target, dirs_exist_ok=True)

    def download_folder(self, path_in_repo: str, local_dir: Path) -> bool:
        source = self.root / path_in_repo
        if not source.is_dir():
            return False
        shutil.copytree(source, local_dir, dirs_exist_ok=True)
        return True

    def url(self) -> str:
        return str(self.root)


class HFHub(Hub):
    def __init__(self, repo_id: str, *, private: bool = True, token: str | None = None) -> None:
        try:
            from huggingface_hub import HfApi
        except ImportError:
            raise RuntimeError("Uploading to Hugging Face needs: pip install huggingface_hub") from None
        token = token or os.environ.get("HF_TOKEN")
        if not token:
            raise RuntimeError("Set HF_TOKEN to a Hugging Face token with write access "
                               "(https://huggingface.co/settings/tokens).")
        self.api = HfApi(token=token)
        self.repo_id = repo_id
        self.api.create_repo(repo_id, private=private, exist_ok=True, repo_type="model")

    def upload_folder(self, local_dir: Path, path_in_repo: str, message: str) -> None:
        self.api.upload_folder(folder_path=str(local_dir), path_in_repo=path_in_repo or None,
                               repo_id=self.repo_id, commit_message=message,
                               delete_patterns=[f"{path_in_repo}/*"] if path_in_repo else None)

    def download_folder(self, path_in_repo: str, local_dir: Path) -> bool:
        from huggingface_hub import snapshot_download

        files = self.api.list_repo_files(self.repo_id)
        if not any(f.startswith(path_in_repo + "/") for f in files):
            return False
        tmp = Path(local_dir).parent / f".hf-download-{os.getpid()}"
        snapshot_download(self.repo_id, allow_patterns=[f"{path_in_repo}/*"], local_dir=str(tmp),
                          token=self.api.token)
        # Move rather than copy: checkpoints of large models are many GB, and disk is limited.
        local_dir = Path(local_dir)
        local_dir.mkdir(parents=True, exist_ok=True)
        for item in (tmp / path_in_repo).iterdir():
            target = local_dir / item.name
            if target.is_dir():
                shutil.rmtree(target)
            elif target.exists():
                target.unlink()
            shutil.move(str(item), str(target))
        shutil.rmtree(tmp, ignore_errors=True)
        return True

    def squash_history(self) -> None:
        """Drop old commits so superseded checkpoints stop using storage."""
        try:
            self.api.super_squash_history(repo_id=self.repo_id)
        except Exception:  # noqa: BLE001 - best effort; an old huggingface_hub may lack it
            pass

    def url(self) -> str:
        return f"https://huggingface.co/{self.repo_id}"
