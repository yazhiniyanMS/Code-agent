"""End-to-end test of `ycode-lm autotrain` across two simulated GPU sessions (CPU, local hub)."""

import json

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("numpy")

from ycode.lm.autotrain import AutoTrainer, Plan  # noqa: E402
from ycode.lm.hub import LocalHub  # noqa: E402
from ycode.lm.train import TrainConfig, load_checkpoint, train  # noqa: E402


@pytest.fixture
def base_model(data_dir, tmp_path):
    cfg = TrainConfig(preset="v4-tiny", model_overrides={"block_size": 64}, batch_size=4, max_steps=10,
                      warmup_steps=2, eval_interval=10, eval_iters=1, log_interval=1000, device="cpu",
                      precision="fp32")
    train(data_dir, tmp_path / "v4", cfg, log=lambda *_: None)
    return tmp_path / "v4"


def _plan(base_model, src, **kw):
    base = dict(base_model=str(base_model), preset="v5-tiny", sources=[str(src)], max_mb=None,
                pretrain_steps=6, sft_steps=4, batch_size=2, nproc=1, precision="fp32",
                grad_checkpoint=False, first_segment_steps=3, eval_interval=3, device="cpu",
                session_hours=5, segment_minutes=60)
    base.update(kw)
    return Plan(**base)


def test_autotrain_resumes_across_sessions_and_publishes(base_model, tmp_path):
    from tests.test_lm import CODE

    src = tmp_path / "src"
    src.mkdir()
    for i in range(6):
        (src / f"m{i}.py").write_text(CODE.replace("add", f"add{i}") * 3)
    ckpt, release = LocalHub(tmp_path / "hub", "you/ckpt"), LocalHub(tmp_path / "hub", "you/ycode-lm-v5")
    logs: list[str] = []

    # Session 1: stops after one segment (as if the 12-hour session ended).
    s1 = AutoTrainer(_plan(base_model, src, max_segments=1), tmp_path / "session1", ckpt, release,
                     log=logs.append).run()
    assert (s1.phase, s1.pretrain_step) == ("pretrain", 3)
    assert any("Grew" in line for line in logs)
    hub_state = json.loads((ckpt.root / "state" / "state.json").read_text())
    assert hub_state["pretrain_step"] == 3 and hub_state["seconds_per_step"] > 0

    # Session 2: a brand-new machine/work dir, so it must resume purely from the hub.
    logs.clear()
    s2 = AutoTrainer(_plan(base_model, src), tmp_path / "session2", ckpt, release, log=logs.append).run()
    assert s2.phase == "done" and s2.pretrain_step == 6 and s2.sft_step == 4 and s2.sessions == 2
    assert not any("Grew" in line for line in logs)  # resumed, did not start over
    assert any("prepared in an earlier session" in line for line in logs)  # data not rebuilt
    assert any("training steps 3 -> 6" in line for line in logs)

    # The published model loads, is version 5, and carries a model card + eval results.
    model, _, payload = load_checkpoint(release.root)
    assert payload["config"]["arch_version"] == 5 and model.cfg.n_embd == 128 and payload["stage"] == "sft"
    card = (release.root / "README.md").read_text()
    assert "YCode-LM v5" in card and "pass@1" in card and "fix@1" in card
    assert json.loads((release.root / "eval.json").read_text())

    # A third run is a no-op.
    s3 = AutoTrainer(_plan(base_model, src), tmp_path / "session3", ckpt, release, log=logs.append).run()
    assert s3.phase == "done"


def test_autotrain_cli_dry_run_requires_repo_or_local_hub(tmp_path, capsys):
    from ycode.lm import cli

    assert cli.main(["autotrain", "--base-model", str(tmp_path / "x")]) == 2
    assert "--hf-repo" in capsys.readouterr().err


def test_hfhub_requires_token(monkeypatch):
    pytest.importorskip("huggingface_hub")
    from ycode.lm.hub import HFHub

    import ycode.lm.hub as hub

    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.setattr(hub, "_saved_login", lambda: None)
    with pytest.raises(RuntimeError, match="HF_TOKEN"):
        HFHub("you/repo")
