from __future__ import annotations

import re
import string
from typing import Any, Iterable, List


_PROFILE_SIGNATURE_SLOTS = {"identity", "research_topic", "education", "occupation"}


def _safe_text(value: Any) -> str:
    return str(value or "").strip()


def _dedupe_texts(items: Iterable[Any]) -> List[str]:
    results: List[str] = []
    seen = set()
    for item in items:
        text = _safe_text(item)
        if not text:
            continue
        key = text.lower()
        if key in seen:
            continue
        seen.add(key)
        results.append(text)
    return results


def _public_lemmatize_token(token: str) -> str:
    lowered = _safe_text(token).lower().strip(string.punctuation)
    if not lowered:
        return ""
    if lowered in {"m", "re", "ve", "ll", "d", "s", "t"}:
        return ""
    if lowered.endswith("ies") and len(lowered) > 4:
        return f"{lowered[:-3]}y"
    if lowered.endswith("ing") and len(lowered) > 5:
        return lowered[:-3]
    if lowered.endswith("ed") and len(lowered) > 4:
        return lowered[:-2]
    if lowered.endswith("s") and len(lowered) > 4 and not lowered.endswith("ss"):
        return lowered[:-1]
    return lowered


def public_content_tokens(text: str, *, speaker: str = "", keep_time_tokens: bool = False) -> List[str]:
    speaker_tokens = {
        _public_lemmatize_token(part)
        for part in re.findall(r"[A-Za-z0-9]+", _safe_text(speaker))
        if _public_lemmatize_token(part)
    }
    tokens: List[str] = []
    for raw in re.findall(r"[A-Za-z0-9\+]+", _safe_text(text)):
        token = _public_lemmatize_token(raw)
        if not token:
            continue
        if token in speaker_tokens:
            continue
        if not keep_time_tokens and token.isdigit():
            continue
        tokens.append(token)
    return _dedupe_texts(tokens)


def _signature_anchor_tokens(text: str, *, speaker: str = "", keep_time_tokens: bool = False, limit: int = 8) -> List[str]:
    raw_tokens = re.findall(r"[A-Za-z0-9\+]+", _safe_text(text))
    if not raw_tokens:
        return []
    speaker_tokens = {
        _public_lemmatize_token(part)
        for part in re.findall(r"[A-Za-z0-9]+", _safe_text(speaker))
        if _public_lemmatize_token(part)
    }
    scored: dict[str, tuple[float, int]] = {}
    total = len(raw_tokens)
    for index, raw in enumerate(raw_tokens):
        token = _public_lemmatize_token(raw)
        if not token or token in speaker_tokens:
            continue
        if not keep_time_tokens and token.isdigit():
            continue
        score = min(3.0, len(token) / 2.5)
        if raw[:1].isupper() and index > 0:
            score += 1.0
        if any(ch.isdigit() for ch in raw) or "+" in raw:
            score += 0.6
        if len(token) <= 2:
            score -= 1.0
        elif len(token) == 3:
            score -= 0.25
        if index >= max(1, total // 2):
            score += 0.2
        current = scored.get(token)
        if current is None or score > current[0]:
            scored[token] = (score, index)
    if not scored:
        return public_content_tokens(text, speaker=speaker, keep_time_tokens=keep_time_tokens)[:limit]
    selected = sorted(scored.items(), key=lambda item: (-item[1][0], item[1][1], item[0]))[: max(1, limit)]
    ordered = sorted(((token, pos) for token, (_, pos) in selected), key=lambda item: item[1])
    return [token for token, _ in ordered]


def compute_public_event_signature(text: str, *, speaker: str = "", semantic_slot: str = "") -> str:
    tokens = _signature_anchor_tokens(text, speaker=speaker)
    normalized_slot = _safe_text(semantic_slot)
    if normalized_slot in _PROFILE_SIGNATURE_SLOTS:
        tokens = _dedupe_texts(
            [
                *_signature_anchor_tokens(normalized_slot.replace("_", " "), keep_time_tokens=True, limit=4),
                *tokens,
            ]
        )
    if len(tokens) <= 8:
        return " ".join(tokens)
    return " ".join(tokens[:8])
