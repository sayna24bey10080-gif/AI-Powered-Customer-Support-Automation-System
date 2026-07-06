"""human_review.py – Determines if a query requires human/supervisor escalation."""

ESCALATION_TRIGGERS = [
    "refund",
    "cancel subscription",
    "account closure",
    "compensation",
    "speak to manager",
    "speak to a manager",
    "escalate",
    "legal",
    "lawsuit",
    "data breach",
    "fraud",
    "unauthorized charge",
    "dispute",
]


def requires_human_review(query: str) -> bool:
    """Return True if the query contains escalation-worthy keywords."""
    q = query.lower()
    return any(trigger in q for trigger in ESCALATION_TRIGGERS)
