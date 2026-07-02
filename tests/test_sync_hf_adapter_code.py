#!/usr/bin/env python3
from __future__ import annotations

import ast
import json
import tempfile
from pathlib import Path

from scripts.sync_hf_adapter_code import ADAPTER_FILES, sync_one


def _converter_adapter_files() -> list[str]:
    script = Path(__file__).resolve().parents[1] / "scripts" / "convert_rwkv7_to_hf.py"
    tree = ast.parse(script.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "copy_adapter_files":
            for child in ast.walk(node):
                if isinstance(child, ast.For) and isinstance(child.iter, ast.List):
                    values = []
                    for item in child.iter.elts:
                        if not isinstance(item, ast.Constant) or not isinstance(item.value, str):
                            break
                        values.append(item.value)
                    else:
                        return values
    raise AssertionError("could not find adapter file list in convert_rwkv7_to_hf.py")


def main() -> int:
    runtime_files = sorted(
        p.name
        for p in (Path(__file__).resolve().parents[1] / "rwkv7_hf").glob("*.py")
        if p.name != "__init__.py"
    )
    assert sorted(ADAPTER_FILES) == runtime_files
    assert _converter_adapter_files() == ADAPTER_FILES

    with tempfile.TemporaryDirectory() as td:
        model_dir = Path(td) / "rwkv7-g1d-0.4b-hf"
        model_dir.mkdir()
        weight = model_dir / "model.safetensors"
        weight.write_bytes(b"do-not-touch")
        (model_dir / "config.json").write_text(
            json.dumps(
                {
                    "architectures": ["OldModel"],
                    "model_type": "old_rwkv7",
                    "auto_map": {"AutoModelForCausalLM": "old.Model"},
                }
            )
            + "\n",
            encoding="utf-8",
        )

        result = sync_one(model_dir)
        assert result["model_dir"] == str(model_dir)
        assert result["dry_run"] is False
        for name in ADAPTER_FILES:
            assert (model_dir / name).exists(), name
        cfg = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
        assert cfg["architectures"] == ["RWKV7ForCausalLM"]
        assert cfg["model_type"] == "rwkv7_hf_adapter"
        assert cfg["auto_map"]["AutoModelForCausalLM"] == "modeling_rwkv7.RWKV7ForCausalLM"
        assert weight.read_bytes() == b"do-not-touch"

    print("PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
