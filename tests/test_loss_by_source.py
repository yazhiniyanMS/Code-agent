"""V13: the trainer logs the answer-token loss of each data source every step (metrics.jsonl), to watch forgetting while
training: the loss on replayed rows against the loss on the new rows."""
import json

import pytest
import torch

from tests.test_v10 import base_model  # noqa: F401 - fixture
from ycode.lm.data import InstructionExample


@pytest.fixture
def tagged_sft(data_dir, tmp_path):
    from ycode.lm.basics import write_basics_dataset

    examples = [InstructionExample(f"Write `f{i}(x)`.", f"```python\ndef f{i}(x):\n    return x + {i}\n```") for i in range(8)]
    tags = [("strings:write:f" if i % 2 else "legacy:write") for i in range(8)] * 3
    write_basics_dataset(tmp_path / "tagged", data_dir / "tokenizer.json", examples=examples * 3, seed=0,
                         log=lambda *_: None, tags=tags)
    return tmp_path / "tagged"


def test_a_training_batch_knows_the_source_of_every_target_token(tagged_sft):
    from ycode.lm.train import SFTData

    gen = torch.Generator().manual_seed(0)
    data = SFTData(tagged_sft, block_size=256, batch_size=1, device="cpu", pad_id=0, pack=False)
    assert data.source_names == ["legacy", "strings"]
    tags = json.loads((tagged_sft / "sft_tags.json").read_text())
    for _ in range(6):  # one example per row: every target token is that example's
        _, _, m = data.batch("train", gen, ordered=True)
        want = data.source_names.index(tags[data._last].split(":", 1)[0])
        assert data.last_sources.shape == m.shape and set(data.last_sources[0].tolist()) == {want}
    packed = SFTData(tagged_sft, block_size=64, batch_size=2, device="cpu", pad_id=0)
    _, _, m = packed.batch("train", gen, ordered=True)
    assert packed.last_sources.shape == m.shape and set(packed.last_sources.unique().tolist()) <= {0, 1}
    packed.batch("val", gen)  # evaluation batches are not tracked
    assert packed.last_sources is None


def test_an_untagged_dataset_tracks_nothing(data_dir, tmp_path):
    from ycode.lm.basics import write_basics_dataset
    from ycode.lm.train import SFTData

    examples = [InstructionExample(f"Write `g{i}(x)`.", f"```python\ndef g{i}(x):\n    return x * {i}\n```") for i in range(8)]
    write_basics_dataset(tmp_path / "plain", data_dir / "tokenizer.json", examples=examples * 3, seed=0,
                         log=lambda *_: None)
    data = SFTData(tmp_path / "plain", block_size=64, batch_size=2, device="cpu", pad_id=0)
    data.batch("train", torch.Generator().manual_seed(0), ordered=True)
    assert data.source_names == [] and data.last_sources is None


def test_training_logs_each_sources_loss_and_they_average_to_the_training_loss(base_model, tagged_sft,  # noqa: F811
                                                                                tmp_path):
    from ycode.lm.train import TrainConfig, train

    out = tmp_path / "ft"
    train(tagged_sft, out, TrainConfig(stage="sft", init_from=str(base_model), max_steps=3, batch_size=2,
                                       warmup_steps=1, eval_iters=0, eval_interval=1000, device="cpu",
                                       precision="fp32", lr=1e-3), log=lambda *_: None)
    rows = [json.loads(line) for line in (out / "metrics.jsonl").read_text().splitlines()]
    assert len(rows) == 3
    for row in rows:
        by, weight = row["loss_by_source"], row["answer_weight_by_source"]
        assert set(by) <= {"legacy", "strings"} and set(by) == set(weight)
        mean = sum(by[k] * weight[k] for k in by) / sum(weight.values())
        assert mean == pytest.approx(row["loss"], rel=1e-4, abs=1e-5)  # the same tokens, weighted the same way
