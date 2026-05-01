from urllib.parse import urlparse, urlunparse

MAX_URL_LENGTH = 2048

BLOCKED_DOMAINS = {
    "evil.com",
    "malware.example.com",
    "phishing.example.com",
}


def is_blocked_domain(hostname: str | None) -> bool:
    if hostname is None:
        return True
    return hostname.lower() in BLOCKED_DOMAINS


def validate_url(url: str) -> str:
    """Format check, normalization, and blocklist validation."""
    #
    # Design decision: normalization keeps the same destination URL mapping to
    # the same token (no duplicates); blocklist validation prevents short links
    # from becoming phishing vectors.
    #
    # Hints:
    # 1. Validate: length within MAX_URL_LENGTH, scheme is http/https via
    #    urlparse(), hostname is not in is_blocked_domain(). Raise ValueError otherwise.
    # 2. Normalize and return: lowercase, strip trailing slash, upgrade http→https.
    candidate = url.strip()
    if not candidate:
        raise ValueError("URL is required.")

    if len(candidate) > MAX_URL_LENGTH:
        raise ValueError(f"URL exceeds maximum length of {MAX_URL_LENGTH} characters.")

    parsed = urlparse(candidate)
    scheme = parsed.scheme.lower()

    if scheme not in ("http", "https"):
        raise ValueError("Invalid URL format: Only http and https schemes are allowed.")

    hostname = parsed.hostname
    if hostname is None:
        raise ValueError("URL must include a valid hostname.")

    if is_blocked_domain(hostname):
        raise ValueError("Security Warning: The provided domain is blocked.")

    normalized_scheme = "https"
    normalized_hostname = hostname.lower()
    port = f":{parsed.port}" if parsed.port is not None else ""
    normalized_netloc = f"{normalized_hostname}{port}"

    path = parsed.path or ""
    if path != "/":
        path = path.rstrip("/")

    return urlunparse(
        (
            normalized_scheme,
            normalized_netloc,
            path,
            parsed.params,
            parsed.query,
            parsed.fragment,
        )
    )
