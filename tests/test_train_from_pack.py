"""A checkpoint pack (ycode.lm.packckpt) can be loaded, evaluated and trained from as it is: no unpacked copy on a RAM
disk first. Training from a pack is the same run, bit for bit, as training from the directory it was packed from."""
import json

import torch

from tests.test_v10 import _finetune, base_model, sft  # noqa: F401 - fixtures


def _weights(path):
    return torch.load(path / "model.pt", map_location="cpu", weights_only=True)


def test_a_pack_loads_like_its_directory(base_model, sft, tmp_path):  # noqa: F811
    from ycode.lm import packckpt
    from ycode.lm.generate import stored_params
    from ycode.lm.train import info_path, load_checkpoint

    first = _finetune(sft, base_model, tmp_path / "first", 1)
    second = _finetune(sft, first, tmp_path / "second", 1)
    packckpt.main(["pack", str(first), str(tmp_path / "first.ypk")])
    packckpt.main(["pack", str(second), str(tmp_path / "second.ypk"), str(tmp_path / "first.ypk")])  # a delta pack
    for directory, pack in ((first, tmp_path / "first.ypk"), (second, tmp_path / "second.ypk")):
        a, b = load_checkpoint(directory)[0].state_dict(), load_checkpoint(pack)[0].state_dict()
        assert a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a)
        assert info_path(pack) == pack.with_suffix(".info.json")
        assert stored_params(pack) == stored_params(directory) is not None  # from the sidecar, without loading
    meta, names = packckpt.describe(tmp_path / "second.ypk")
    assert meta["base"] == _weights(second)["base"] and sorted(names) == sorted(_weights(second)["model"])


def test_training_from_a_pack_is_the_same_run_as_from_its_directory(base_model, sft, tmp_path):  # noqa: F811
    from ycode.lm import packckpt

    first = _finetune(sft, base_model, tmp_path / "first", 1)
    packed = tmp_path / "first.ypk"
    packckpt.main(["pack", str(first), str(packed)])
    from_dir = _finetune(sft, first, tmp_path / "from_dir", 1)
    from_pack = _finetune(sft, packed, tmp_path / "from_pack", 1)
    a, b = _weights(from_dir), _weights(from_pack)
    assert a["base"] == b["base"]  # both reference the plain base, as a run from the directory does
    assert a["model"].keys() == b["model"].keys()
    assert all(torch.equal(a["model"][k], b["model"][k]) for k in a["model"])
    losses = [[json.loads(line)["loss"] for line in (d / "metrics.jsonl").read_text().splitlines()]
              for d in (from_dir, from_pack)]
    assert losses[0] == losses[1]
    summaries = [json.loads((d / "train_log.json").read_text())["summary"] for d in (from_dir, from_pack)]
    assert summaries[0]["init_sha256"] == summaries[1]["init_sha256"] is not None  # read from the pack's sidecar


def test_the_sft_command_starts_from_a_pack(base_model, sft, tmp_path):  # noqa: F811
    from ycode.lm import packckpt
    from ycode.lm.cli import main

    first = _finetune(sft, base_model, tmp_path / "first", 1)
    packckpt.main(["pack", str(first), str(tmp_path / "first.ypk")])
    code = main(["sft", "--data", str(sft), "--init-from", str(tmp_path / "first.ypk"), "--out", str(tmp_path / "cli"),
                 "--steps", "1", "--batch-size", "2", "--device", "cpu", "--precision", "fp32", "--eval-iters", "0",
                 "--train-layers", "1", "--compact-checkpoints"])
    assert code in (0, None) and (tmp_path / "cli" / "model.pt").is_file()
    assert _weights(tmp_path / "cli")["base"] == _weights(first)["base"]
