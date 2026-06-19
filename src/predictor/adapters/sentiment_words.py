"""
src.predictor.adapters.sentiment_words
--------------------------------------
Keyword wordlists for headline scoring (docs/PREDICTOR.md §3 / DESIGN.md Phase 1).
No LLM calls in the pipeline — sentiment is transparent keyword matching, which is
cheap, deterministic, and auditable. Order-wins are their own first-class class.

Matching is done on lowercased headline text; multi-word phrases match as substrings.
"""

# Positive / bullish catalysts
POSITIVE = [
    "surge", "jumps", "rally", "soars", "record high", "all-time high", "upgrade",
    "raises target", "price target", "outperform", "buy rating", "beats estimates",
    "profit rises", "profit jumps", "revenue growth", "strong results", "expansion",
    "capacity addition", "new plant", "stake buy", "promoter buying", "bonus issue",
    "dividend", "buyback", "fundraise", "approval", "clearance", "partnership", "tie-up",
    "acquisition", "margin expansion", "guidance raised", "multibagger", "breakout",
]

# Negative / bearish catalysts
NEGATIVE = [
    "plunge", "slumps", "crashes", "tumbles", "downgrade", "cuts target", "sell rating",
    "underperform", "misses estimates", "profit falls", "loss widens", "revenue decline",
    "weak results", "probe", "investigation", "fraud", "default", "downgraded", "fine",
    "penalty", "raid", "resignation", "promoter pledge", "stake sale", "block deal sell",
    "lawsuit", "recall", "ban", "halt", "insolvency", "NCLT", "debt concern", "guidance cut",
]

# Order-win / order-book events — first-class class (DESIGN.md §1, requirement)
ORDER_WIN = [
    "bags order", "wins order", "secures order", "order win", "order worth", "contract worth",
    "bags contract", "wins contract", "secures contract", "new order", "order book",
    "loi", "letter of intent", "work order", "awarded", "bagged", "emerges lowest bidder",
    "l1 bidder", "purchase order", "supply order", "export order", "deal worth",
]


def score_headline(text: str) -> dict:
    """Return {pos, neg, order_win} keyword hit counts for one lowercased headline."""
    t = (text or "").lower()
    return {
        "pos": sum(1 for w in POSITIVE if w in t),
        "neg": sum(1 for w in NEGATIVE if w in t),
        "order_win": 1 if any(w in t for w in ORDER_WIN) else 0,
    }
