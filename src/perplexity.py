"""Perplexity-based collapse guard. Compute PPL on a fixed held-out corpus
under an arbitrary context manager (steered or not)."""
from __future__ import annotations
import math, torch
from contextlib import nullcontext
from pathlib import Path

from config import DATA_DIR
from model_loader import LoadedModel

# A short neutral passage. Hardcoded so the experiment has no external dep.
_CORPUS = """The city library opened at nine in the morning. A thin stream of patrons came in to return books and browse the new arrivals shelf near the front entrance. The reference desk was staffed by two librarians who were comparing notes on an exhibit they were planning for the following month. Outside, the square was quiet. A vendor selling roasted nuts set up his cart in the usual corner. The temperature was mild and the light was clear, with the particular sharpness that comes after a long cold front moves east. In a nearby coffee shop, a regular ordered a small drip coffee and a plain croissant and read the newspaper for twenty minutes before walking back to her apartment. The city operates by thousands of such routines, most of them invisible to anyone who is not currently inside one. Later in the morning a delivery truck would arrive at the library and unload boxes of interlibrary loans, and the afternoon would bring a small reading group for children. None of this required special attention from anyone passing by. It is the ordinary and largely uncelebrated work of a municipality, performed by people who arrive on time, know their small part of the whole, and move through their day with the quiet competence that makes such mornings possible."""


def corpus_path() -> Path:
    p = DATA_DIR / "perplexity_corpus.txt"
    if not p.exists():
        p.write_text(_CORPUS, encoding="utf-8")
    return p


@torch.no_grad()
def perplexity(lm: LoadedModel, cm=None, max_tokens: int = 512) -> float:
    text = corpus_path().read_text(encoding="utf-8")
    enc = lm.tokenizer(text, return_tensors="pt").to(lm.model.device)
    ids = enc["input_ids"][:, :max_tokens]
    if cm is None:
        cm = nullcontext()
    with cm:
        out = lm.model(ids, labels=ids)
    return math.exp(float(out.loss))
