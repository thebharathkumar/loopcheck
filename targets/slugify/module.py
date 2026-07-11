import re
import unicodedata


def slugify(text: str, max_length: int = 64) -> str:
    """Convert text to a URL-safe slug. See SPEC.md."""
    if max_length < 1:
        raise ValueError("max_length must be >= 1")
    normalized = unicodedata.normalize("NFKD", text)
    ascii_text = normalized.encode("ascii", "ignore").decode("ascii")
    cleaned = re.sub(r"[^a-z0-9]+", "-", ascii_text.lower()).strip("-")
    if len(cleaned) > max_length:
        cleaned = cleaned[:max_length].rstrip("-")
    return cleaned
