"""Train and use a Byte-Level BPE tokenizer from PaoPao data only."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Sequence

SPECIAL_TOKENS = [
    "<pad>",
    "<bos>",
    "<eos>",
    "<unk>",
    "<|task:generate|>",
    "<|task:complete|>",
    "<|task:explain|>",
    "<|task:detect|>",
    "<|task:bugfix|>",
    "<|code|>",
    "<|error|>",
    "<|test_failure|>",
    "<|explanation|>",
    "<fim_prefix>",
    "<fim_suffix>",
    "<fim_middle>",
]


def _require_tokenizers() -> None:
    try:
        import tokenizers  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "The `tokenizers` package is required. Run `pip install -r requirements.txt`."
        ) from exc


class PaoPaoTokenizer:
    """A code-oriented Byte-Level BPE tokenizer trained from local corpora.

    This wrapper never downloads or loads a pretrained vocabulary.  Its vocabulary,
    merge rules, and special-token IDs are produced by :meth:`train`.
    """

    def __init__(self, backend: object, special_tokens: Sequence[str] = SPECIAL_TOKENS) -> None:
        self._backend = backend
        self.special_tokens = list(special_tokens)
        self.pad_token_id = self.token_to_id("<pad>")
        self.bos_token_id = self.token_to_id("<bos>")
        self.eos_token_id = self.token_to_id("<eos>")
        self.unk_token_id = self.token_to_id("<unk>")
        if any(value is None for value in (self.pad_token_id, self.bos_token_id, self.eos_token_id)):
            raise ValueError("tokenizer is missing required special tokens")

    @property
    def vocab_size(self) -> int:
        """Number of tokens in the locally trained vocabulary."""
        return self._backend.get_vocab_size()

    @classmethod
    def train(
        cls,
        texts: Iterable[str],
        vocab_size: int = 32_000,
        min_frequency: int = 2,
        special_tokens: Sequence[str] = SPECIAL_TOKENS,
    ) -> "PaoPaoTokenizer":
        """Train a fresh Byte-Level BPE model over supplied source-code text."""
        _require_tokenizers()
        from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

        if vocab_size <= len(special_tokens):
            raise ValueError("vocab_size must exceed the number of special tokens")
        backend = Tokenizer(models.BPE(unk_token="<unk>"))
        backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        backend.decoder = decoders.ByteLevel()
        trainer = trainers.BpeTrainer(
            vocab_size=vocab_size,
            min_frequency=min_frequency,
            special_tokens=list(special_tokens),
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        )
        backend.train_from_iterator(texts, trainer=trainer, length=None)
        return cls(backend, special_tokens)

    @classmethod
    def from_file(cls, directory: str | Path) -> "PaoPaoTokenizer":
        """Load a tokenizer previously saved by :meth:`save`."""
        _require_tokenizers()
        from tokenizers import Tokenizer

        directory = Path(directory)
        config_path = directory / "tokenizer_config.json"
        tokenizer_path = directory / "tokenizer.json"
        if not tokenizer_path.is_file() or not config_path.is_file():
            raise FileNotFoundError(f"expected tokenizer.json and tokenizer_config.json in {directory}")
        settings = json.loads(config_path.read_text(encoding="utf-8"))
        return cls(Tokenizer.from_file(str(tokenizer_path)), settings["special_tokens"])

    def save(self, directory: str | Path) -> None:
        """Persist all artifacts needed to reload this trained tokenizer."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self._backend.save(str(directory / "tokenizer.json"))
        (directory / "tokenizer_config.json").write_text(
            json.dumps({"special_tokens": self.special_tokens}, indent=2) + "\n", encoding="utf-8"
        )

    def token_to_id(self, token: str) -> int | None:
        """Get a vocabulary ID for a token."""
        return self._backend.token_to_id(token)

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        """Encode text while preserving whitespace and source-code punctuation."""
        ids = list(self._backend.encode(text).ids)
        if add_special_tokens:
            return [int(self.bos_token_id), *ids, int(self.eos_token_id)]
        return ids

    def decode(self, ids: Sequence[int], skip_special_tokens: bool = False) -> str:
        """Decode IDs to source text."""
        return self._backend.decode(list(ids), skip_special_tokens=skip_special_tokens)
