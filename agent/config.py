"""Configuration and the model client.

A six-line .env parser instead of python-dotenv: its find_dotenv() walks the
call stack and breaks when the entry point is stdin or a frozen script, which
is exactly the kind of failure you do not want at 2am.
"""
from __future__ import annotations

import os
import ssl
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BASE_URL = "https://api.tensormux.com/v1"
MODEL = "glm-4-7-flash"


def load_env(path: Path | None = None) -> None:
    """Read KEY=value lines into os.environ. Existing vars win."""
    env_file = path or (ROOT / ".env")
    if not env_file.exists():
        return
    # utf-8-sig: Notepad and PowerShell both like to leave a BOM.
    for line in env_file.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip('"').strip("'")
        if key and value and key not in os.environ:
            os.environ[key] = value


def api_key() -> str | None:
    load_env()
    key = os.environ.get("TENSORMUX_API_KEY", "").strip()
    if not key or key == "PASTE_YOUR_KEY_HERE":
        return None
    return key


def _ssl_context() -> ssl.SSLContext | None:
    """Trust the OS certificate store, not just certifi's bundle.

    Consumer antivirus (Avast, Kaspersky, ESET) terminates TLS locally and
    re-signs with its own root, which lives in the Windows store. Python ships
    certifi and therefore rejects it, while curl and the browser are fine --
    so the network looks healthy and only Python fails.
    """
    try:
        import truststore

        return truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    except ImportError:
        return None


@lru_cache(maxsize=1)
def client():
    """The model client, or None when no key is configured.

    Returning None rather than raising is deliberate: the pipeline has a local
    fallback and must stay runnable with no credentials at all.
    """
    key = api_key()
    if key is None:
        return None
    import httpx
    from openai import OpenAI

    ctx = _ssl_context()
    http_client = httpx.Client(verify=ctx) if ctx else None
    return OpenAI(
        api_key=key,
        base_url=BASE_URL,
        timeout=60.0,
        max_retries=2,
        http_client=http_client,
    )
