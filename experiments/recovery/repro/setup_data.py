"""Build the experiment's inputs once: a tokenizer + pretraining data (from the test suite's source text) and an
SFT dataset. Prints the inputs' hashes so a rerun can be checked against them."""
import hashlib, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # the repository root
from tests.test_lm import CODE
from ycode.lm.data import prepare_dataset
from ycode.lm.basics import write_basics_dataset
from ycode.lm.data import InstructionExample

root = Path(sys.argv[1]); src = root / "src"; src.mkdir(parents=True)
for i in range(6):
    (src / f"m{i}.py").write_text(CODE.replace("add", f"add{i}") * 3)
prepare_dataset([src], root / "data", vocab_size=400, workers=1, log=lambda *_: None)
ex = [InstructionExample(f"Write `f{i}(x)`.", f"```python\ndef f{i}(x):\n    return x + {i}\n```") for i in range(8)]
write_basics_dataset(root / "sft", root / "data" / "tokenizer.json", examples=ex * 3, seed=0, log=lambda *_: None)
for p in sorted(list((root / "data").iterdir()) + list((root / "sft").iterdir())):
    if p.is_file():
        print(p.relative_to(root), hashlib.sha256(p.read_bytes()).hexdigest()[:16])
