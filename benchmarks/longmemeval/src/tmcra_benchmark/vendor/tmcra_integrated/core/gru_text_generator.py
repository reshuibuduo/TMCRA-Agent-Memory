"""
GRU-based character generator (self-trained, no external embeddings/LLM).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Iterable, Optional

try:
    import torch
    from torch import nn
except Exception:  # pragma: no cover - optional runtime dependency
    torch = None
    nn = None


SPECIAL_TOKENS = ["<PAD>", "<BOS>", "<EOS>", "<PATH>", "<SEP>"]


@dataclass
class GRUConfig:
    embed_dim: int = 128
    hidden_dim: int = 256
    num_layers: int = 2
    dropout: float = 0.1


class CharVocab:
    def __init__(self, tokens: List[str]):
        self.tokens = tokens
        self.token_to_id = {t: i for i, t in enumerate(tokens)}
        self.id_to_token = {i: t for i, t in enumerate(tokens)}

    @classmethod
    def build(cls, corpus_iter: Iterable[str]) -> "CharVocab":
        chars = set()
        for text in corpus_iter:
            chars.update(list(text))
        tokens = SPECIAL_TOKENS + sorted(chars)
        return cls(tokens)

    def encode(self, text: str) -> List[int]:
        return [self.token_to_id.get(ch, self.token_to_id["<PAD>"]) for ch in text]

    def decode(self, ids: Iterable[int]) -> str:
        return "".join(self.id_to_token.get(i, "") for i in ids)


if nn is not None:
    class GRUCharModel(nn.Module):
        def __init__(self, vocab_size: int, config: GRUConfig):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, config.embed_dim)
            self.gru = nn.GRU(
                input_size=config.embed_dim,
                hidden_size=config.hidden_dim,
                num_layers=config.num_layers,
                dropout=config.dropout if config.num_layers > 1 else 0.0,
                batch_first=True,
            )
            self.fc = nn.Linear(config.hidden_dim, vocab_size)

        def forward(self, x, hidden=None):
            emb = self.embedding(x)
            out, hidden = self.gru(emb, hidden)
            logits = self.fc(out)
            return logits, hidden
else:  # pragma: no cover - fallback when torch is unavailable
    class GRUCharModel:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("torch not available; install torch to use GRUCharModel")


class GRUTextGenerator:
    def __init__(self, vocab: CharVocab, config: GRUConfig | None = None):
        if torch is None:
            raise RuntimeError("torch not available; install torch to use GRUTextGenerator")
        self.vocab = vocab
        self.config = config or GRUConfig()
        self.model = GRUCharModel(len(vocab.tokens), self.config)

    @classmethod
    def load(cls, model_path: str | Path) -> "GRUTextGenerator":
        if torch is None:
            raise RuntimeError("torch not available; install torch to load GRUTextGenerator")
        data = torch.load(str(model_path), map_location="cpu", weights_only=True)
        vocab = CharVocab(data["vocab"])
        config = GRUConfig(**data["config"])
        gen = cls(vocab, config)
        gen.model.load_state_dict(data["state_dict"])
        gen.model.eval()
        return gen

    def save(self, model_path: str | Path) -> None:
        data = {
            "vocab": self.vocab.tokens,
            "config": self.config.__dict__,
            "state_dict": self.model.state_dict(),
        }
        Path(model_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(data, str(model_path))

    def _prefix_ids(self, path_text: str) -> List[int]:
        prefix = "<PATH>" + path_text + "<SEP>"
        return [self.vocab.token_to_id["<BOS>"]] + self.vocab.encode(prefix)

    def generate(
        self,
        path_text: str,
        max_len: int = 400,
        temperature: float = 0.9,
        device: str = "cpu",
    ) -> str:
        self.model.to(device)
        self.model.eval()
        ids = self._prefix_ids(path_text)
        input_ids = torch.tensor([ids], dtype=torch.long, device=device)
        hidden = None
        generated: List[int] = []
        for _ in range(max_len):
            logits, hidden = self.model(input_ids, hidden)
            next_logits = logits[:, -1, :] / max(temperature, 1e-6)
            probs = torch.softmax(next_logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1).item()
            if next_id == self.vocab.token_to_id["<EOS>"]:
                break
            generated.append(next_id)
            input_ids = torch.tensor([[next_id]], dtype=torch.long, device=device)
        return self.vocab.decode(generated)

    @staticmethod
    def build_corpus_iter(corpus_path: str | Path, max_chars: int | None = None) -> Iterable[str]:
        total = 0
        with open(corpus_path, "r", encoding="utf-8", errors="ignore") as handle:
            for line in handle:
                if not line.strip():
                    continue
                if max_chars and total >= max_chars:
                    break
                total += len(line)
                yield line.strip()
