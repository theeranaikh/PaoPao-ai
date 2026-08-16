from __future__ import annotations

import pytest

def test_tokenizer_train_save_load(tmp_path):
    pytest.importorskip("tokenizers")
    from tokenizer import PaoPaoTokenizer

    tokenizer = PaoPaoTokenizer.train(
        ["def hello(name):\n    return f'Hello {name}'\n# comment\n", "x = 1 + 2\n"],
        vocab_size=300,
        min_frequency=1,
    )
    tokenizer.save(tmp_path)
    loaded = PaoPaoTokenizer.from_file(tmp_path)
    ids = loaded.encode("if x == 1:\n    print('yes')\n", add_special_tokens=True)
    assert ids[0] == loaded.bos_token_id
    assert ids[-1] == loaded.eos_token_id
    assert loaded.vocab_size == tokenizer.vocab_size


def test_cli_parser_smoke():
    from cli.main import _normalize_direct_prompt, build_parser, main

    parser = build_parser()
    arguments = parser.parse_args(["--root", ".", "run", "main.py"])
    assert arguments.command == "run"
    assert arguments.file == "main.py"
    assert _normalize_direct_prompt(["--checkpoint", "model.pt", "write code"]) == [
        "--checkpoint", "model.pt", "generate", "write code"
    ]


def test_cli_run_smoke_without_model(tmp_path):
    from cli.main import main

    file_path = tmp_path / "hello.py"
    file_path.write_text("print('hello')\n", encoding="utf-8")
    assert main(["--root", str(tmp_path), "run", "hello.py"]) == 0


def test_prepare_cli_maps_output_to_output_dir(monkeypatch):
    pytest.importorskip("torch")
    pytest.importorskip("tokenizers")
    from data import prepare as prepare_module

    captured = {}

    def fake_prepare(
        inputs,
        tokenizer_path,
        output_dir,
        sequence_length,
        validation_ratio,
        fim_probability,
        completion_probability,
        seed,
        force=False,
    ):
        captured.update(locals())
        return {"examples": 1}

    monkeypatch.setattr(prepare_module, "prepare", fake_prepare)

    assert prepare_module.main(
        [
            "--input",
            "data/raw/python",
            "repairs.jsonl",
            "--tokenizer",
            "artifacts/tokenizer",
            "--output",
            "data/custom",
        ]
    ) == 0
    assert captured["inputs"] == ["data/raw/python", "repairs.jsonl"]
    assert captured["tokenizer_path"] == "artifacts/tokenizer"
    assert captured["output_dir"] == "data/custom"
