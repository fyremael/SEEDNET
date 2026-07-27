from pathlib import Path
import compileall
import json

root = Path(__file__).resolve().parents[1]
assert compileall.compile_dir(root / "src", quiet=1)
assert compileall.compile_dir(root / "tests", quiet=1)
json.loads((root / "notebooks" / "seednet_demo.ipynb").read_text())
print("Static project checks passed.")
