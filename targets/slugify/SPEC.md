# slugify(text, max_length=64) -> str

Converts arbitrary text into a URL-safe slug.

- Unicode is NFKD-normalized and non-ASCII characters dropped ("Crème" → "creme"; emoji vanish).
- Result is lowercase; every run of non-alphanumeric characters becomes a single hyphen.
- No leading or trailing hyphens, even after truncation.
- Truncated to `max_length` characters; a hyphen left dangling by truncation is removed.
- Empty or all-symbol input returns "".
- `max_length < 1` raises ValueError.
