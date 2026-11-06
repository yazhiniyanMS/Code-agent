"""`ycode-lm hardware`: the machine report, device checks, benchmarks and recommendation (ycode.lm.hardware).

The macOS paths (sysctl, vm_stat, memory_pressure) run here on captured output, so they are tested without a
Mac; the MPS device itself can only be checked on one."""
import json
import sys

import pytest

from tests.test_v10 import base_model  # noqa: F401 - fixture
from ycode.lm import hardware

VM_STAT = """Mach Virtual Memory Statistics: (page size of 16384 bytes)
Pages free:                               100000.
Pages active:                             500000.
Pages inactive:                           200000.
Pages speculative:                         50000.
Pages throttled:                               0.
Pages wired down:                         300000.
Pages purgeable:                           10000.
"""
MACOS = {("sysctl", "-n", "machdep.cpu.brand_string"): "Apple M3 Pro",
         ("sysctl", "-n", "hw.perflevel0.physicalcpu"): "6", ("sysctl", "-n", "hw.perflevel1.physicalcpu"): "6",
         ("sysctl", "-n", "hw.memsize"): "38654705664", ("vm_stat",): VM_STAT,
         ("memory_pressure", "-Q"): "The system has 38654705664 (2359296 pages with a page size of 16384).\n"
                                    "System-wide memory free percentage: 63%",
         ("sysctl", "-n", "hw.optional.arm.FEAT_BF16"): "1"}


def test_memory_parsers():
    meminfo = "MemTotal:       16384000 kB\nMemFree:  100 kB\nMemAvailable:    8192000 kB\n"
    assert hardware.parse_meminfo(meminfo) == {"total_gb": 16.384, "available_gb": 8.192}
    assert hardware.parse_vm_stat(VM_STAT) == pytest.approx((100000 + 200000 + 50000 + 10000) * 16384 / 1e9)
    assert hardware.parse_vm_stat("no page size here") is None
    assert hardware.parse_memory_pressure(MACOS[("memory_pressure", "-Q")]) == 63


def test_the_machine_report_here(tmp_path):
    m = hardware.machine(tmp_path)
    assert m["cores"] >= 1 and m["disk_free_gb"] > 0 and m["torch"] and m["python"]
    if sys.platform.startswith("linux"):
        assert not m["apple_silicon"] and m["memory_total_gb"] > m["memory_available_gb"] > 0


def test_the_machine_report_on_an_apple_silicon_mac(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setattr(hardware.platform, "machine", lambda: "arm64")
    monkeypatch.setattr(hardware, "_run", lambda cmd: MACOS.get(tuple(cmd)))
    m = hardware.machine(tmp_path)
    assert m["apple_silicon"] and m["cpu"] == "Apple M3 Pro" and m["cpu_bf16"]
    assert (m["performance_cores"], m["efficiency_cores"], m["memory_free_percent"]) == (6, 6, 63)
    assert m["memory_total_gb"] == pytest.approx(38.65, abs=0.01)
    assert m["memory_available_gb"] == pytest.approx(5.90, abs=0.01)
    monkeypatch.setattr(hardware, "_run", lambda cmd: None)  # a command missing or failing: unknown, not an error
    m = hardware.machine(tmp_path)
    assert m["cpu"] is None and m["memory_total_gb"] is None and m["cpu_bf16"] is False


def test_devices_and_checks():
    found = hardware.devices()
    assert found["cpu"]["available"] and set(found) == {"cpu", "mps", "cuda"}
    if sys.platform.startswith("linux"):
        assert not found["mps"]["available"]
    check = hardware.check_device("cpu")
    assert check["ok"] and check["loss_rel_diff"] == 0 and check["grad_rel_diff"] == 0
    bad = hardware.check_device("no-such-device")  # a device that cannot run is reported, not raised
    assert not bad["ok"] and "error" in bad


@pytest.mark.parametrize("train", [True, False])
def test_a_benchmark_measures_or_says_why_not(train):
    row = hardware.benchmark("cpu", "fp32", preset="tiny", train=train, steps=1, gen_tokens=4)
    assert row["tokens_per_s"] > 0 and row["estimated_gb"] >= 0 and "error" not in row
    skipped = hardware.benchmark("cpu", "fp32", preset="tiny", train=train, steps=1, available_gb=1e-6)
    assert "tokens_per_s" not in skipped and skipped["skipped"].startswith("needs ~")


def test_a_benchmark_of_a_checkpoint_training_its_top_layer(base_model):  # noqa: F811
    full = hardware.benchmark("cpu", "fp32", model_dir=base_model, steps=1)
    top = hardware.benchmark("cpu", "fp32", model_dir=base_model, train_layers=1, steps=1)
    assert top["tokens_per_s"] > 0 and top["estimated_gb"] < full["estimated_gb"]
    assert hardware.benchmark("cpu", "bf16", model_dir=base_model, train=False, gen_tokens=4)["tokens_per_s"] > 0


def test_the_recommendation_is_the_fastest_checked_configuration():
    checks = {"cpu": {"ok": True}, "mps": {"ok": False, "error": "differs from the CPU"}}
    rows = [{"task": "train", "device": "cpu", "precision": "fp32", "model": "small", "tokens_per_s": 100.0},
            {"task": "train", "device": "cpu", "precision": "bf16", "model": "small", "tokens_per_s": 250.0},
            {"task": "train", "device": "mps", "precision": "fp32", "model": "small", "tokens_per_s": 900.0},
            {"task": "inference", "device": "cpu", "precision": "bf16", "model": "small", "skipped": "memory"}]
    best = hardware.recommend(checks, rows)
    assert best == {"train": {"device": "cpu", "precision": "bf16", "tokens_per_s": 250.0, "model": "small"}}


def test_the_hardware_command_writes_its_report(tmp_path, capsys):
    from ycode.lm.cli import main

    out = tmp_path / "hardware.json"
    assert main(["hardware", "--bench", "--preset", "tiny", "--steps", "1", "--json", str(out)]) == 0
    report = json.loads(out.read_text())
    assert set(report) == {"machine", "devices", "checks", "benchmarks", "recommendation"}
    assert report["recommendation"]["train"]["device"] == "cpu"
    assert "recommended for train: cpu" in capsys.readouterr().out


def test_inference_reports_load_first_token_decoding_and_memory():
    row = hardware.benchmark("cpu", "fp32", preset="tiny", train=False, gen_tokens=4, prompt_tokens=8)
    assert row["synthetic"] and row["ttft_s"] > 0 and row["tokens_per_s"] > 0 and row["load_s"] >= 0
    assert row["prompt_tokens"] == 8 and row["peak_rss_gb"] > 0
    if sys.platform.startswith("linux"):
        assert 0 < row["rss_gb"] <= row["peak_rss_gb"] + 0.01


def test_a_checkpoint_too_big_for_memory_is_never_loaded(base_model, monkeypatch):  # noqa: F811
    from ycode.lm import train

    def never(*a, **k):
        raise AssertionError("loaded a checkpoint the memory check had refused")

    monkeypatch.setattr(train, "load_checkpoint", never)
    row = hardware.benchmark("cpu", "fp32", model_dir=base_model, train=False, available_gb=1e-6)
    assert row["skipped"].startswith("needs ~") and not row["synthetic"] and row["params"] > 0


def test_a_real_checkpoint_loads_for_inference_in_16_bit(base_model):  # noqa: F811
    row = hardware.benchmark("cpu", "bf16", model_dir=base_model, train=False, gen_tokens=3, prompt_tokens=4)
    assert row["plan"]["actual"] == "bf16" and row["load_s"] > 0 and row["tokens_per_s"] > 0


def test_the_save_benchmark_times_a_save_and_keeps_the_disk_buffer(tmp_path):
    row = hardware.save_benchmark(tmp_path, preset="tiny", min_free_gb=0)
    assert row["seconds"] > 0 and row["bytes"] > 0 and list(tmp_path.iterdir()) == []  # its folder is removed
    full = hardware.save_benchmark(tmp_path, preset="tiny", min_free_gb=1e9)
    assert "seconds" not in full and "GB kept free" in full["skipped"] and list(tmp_path.iterdir()) == []
