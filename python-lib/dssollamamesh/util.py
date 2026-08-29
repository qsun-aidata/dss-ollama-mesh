"""Small helpers with no third-party dependencies."""

import base64
from urllib.parse import urlsplit

# Header bytes are enough to tell the formats Ollama vision models accept apart.
_IMAGE_SIGNATURES = (
    (b"\x89PNG\r\n\x1a\n", "image/png"),
    (b"\xff\xd8\xff", "image/jpeg"),
    (b"GIF87a", "image/gif"),
    (b"GIF89a", "image/gif"),
    (b"BM", "image/bmp"),
)

# What Dataiku's own templates assume when nothing better is known.
DEFAULT_IMAGE_MIME = "image/jpeg"


def get_any(d, *names):
    """Return the first non-empty value for any of the given keys.

    Dataiku and OpenAI use slightly different field names (callId vs tool_call_id);
    this helper keeps message conversion tolerant of both.
    """
    for name in names:
        value = d.get(name)
        if value:
            return value
    return None


def get_first_set(d, *names):
    """Return the first value that was actually set, keeping falsy ones.

    Unlike get_any, ``0``, ``False`` and ``""`` count as set. Generation settings
    need that distinction: ``temperature=0`` is the most common setting there is,
    and get_any would drop it as if it had never been configured.
    """
    for name in names:
        value = d.get(name)
        if value is not None:
            return value
    return None


def guess_image_mime(b64_data):
    """Sniff an image MIME type from the start of a base64 payload.

    DSS does not pass a MIME type with IMAGE_INLINE parts, and labelling a PNG as
    JPEG trips up the stricter vision models. Decoding the first 18 bytes is enough
    for every signature we check; anything unrecognized keeps the jpeg default.
    """
    head = (b64_data or "")[:24]
    head = head[: len(head) - len(head) % 4]
    if not head:
        return DEFAULT_IMAGE_MIME
    try:
        raw = base64.b64decode(head)
    except Exception:
        return DEFAULT_IMAGE_MIME

    for signature, mime in _IMAGE_SIGNATURES:
        if raw.startswith(signature):
            return mime
    if raw[:4] == b"RIFF" and raw[8:12] == b"WEBP":
        return "image/webp"
    return DEFAULT_IMAGE_MIME


def validate_image_uri(uri):
    """Allow only HTTP(S) image URIs before passing them to Ollama.

    Ollama or a proxy may fetch an image URI server-side, so schemes such as
    ``file:``, ``gopher:`` and ``ftp:`` must not cross this boundary. Host
    allowlisting remains an operator concern because private HTTP(S) image
    endpoints are valid in many DSS deployments.
    """
    if not isinstance(uri, str) or not uri:
        raise ValueError("IMAGE_URI imageUrl must be a non-empty HTTP(S) URL")
    try:
        parsed = urlsplit(uri)
        port = parsed.port
    except ValueError as err:
        raise ValueError("IMAGE_URI imageUrl is malformed") from err
    valid_port = port is None or 1 <= port <= 65535
    if parsed.scheme not in ("http", "https") or not parsed.hostname or not valid_port:
        raise ValueError("IMAGE_URI imageUrl must be an HTTP(S) URL")
    if parsed.username is not None or parsed.password is not None or "@" in parsed.netloc:
        raise ValueError("IMAGE_URI imageUrl must not contain embedded credentials")
    if parsed.fragment:
        raise ValueError("IMAGE_URI imageUrl must not contain a fragment")
    return uri
