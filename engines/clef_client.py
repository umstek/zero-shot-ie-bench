"""Minimal client for Cloudflare's Clef decision models on Workers AI.

Wire format is the TypeSafe System One shape (the one engines/jev_client.py
implements) behind Cloudflare's REST v4 envelope: POST {model, state,
questions} to
https://api.cloudflare.com/client/v4/accounts/<account>/ai/run/@cf/cloudflare/<model>
and the payload arrives wrapped as {"success": true, "result": {model,
answers, usage}} — jev_client.ask() unwraps a "result" envelope when
"answers" is not top-level. Questions and answers are Jev-compatible;
choice criteria descriptions may be null.

Credentials — no dashboard token needed:
- CLOUDFLARE_ACCOUNT_ID from the environment or repo-root .env; absent
  that, the first account from `cf auth whoami`.
- Bearer token, first match wins:
  1. CLOUDFLARE_AUTH_TOKEN (env or .env) — an API token with Workers AI
     permissions, if you create one in the dash. A cf OAuth login cannot
     mint API tokens (403), so this stays a manual dash step.
  2. The `cf` CLI's OAuth session token (config at
     %APPDATA%/xdg.config/cloudflare/config/default.json on Windows,
     ~/.config/cloudflare/config/default.json elsewhere). It lives ~1 h;
     load_token() refreshes it via `cf auth whoami` when under 120 s
     remain (any cf command refreshes it). `cf auth login` once, then
     the benches just work.

Cloudflare states Workers AI does not read, store, or train on Clef
requests (unlike the non-ZDR OpenRouter systems); input bills at
$0.24/M tokens (a 128-token probe ~= $0.00003).

No third-party deps on purpose: urllib only.
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from datetime import datetime, timezone

# body "model" selector -> endpoint model id
MODELS = {
    "clef": "@cf/cloudflare/clef",          # Qwen3.8-27B, 64k ctx, vision
    "clef-flash": "@cf/cloudflare/clef-flash",  # Qwen3.5-9B, ~39 ms median
}
API_BASE = "https://api.cloudflare.com/client/v4"


def _env_value(name: str) -> str:
    env = os.environ.get(name)
    if env:
        return env.strip()
    env_path = os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), ".env")
    try:
        with open(env_path, encoding="utf-8") as fh:
            for line in fh:
                if line.startswith(f"{name}="):
                    return line.split("=", 1)[1].strip()
    except OSError:
        pass
    return ""


def _cf_config_path() -> str:
    appdata = os.environ.get("APPDATA")
    if appdata:
        return os.path.join(appdata, "xdg.config", "cloudflare",
                            "config", "default.json")
    return os.path.join(os.path.expanduser("~"), ".config", "cloudflare",
                        "config", "default.json")


def _read_cf_session() -> tuple[str, float]:
    with open(_cf_config_path(), encoding="utf-8") as fh:
        cfg = json.load(fh)
    # malformed shapes raise ValueError so load_token()'s error boundary
    # turns them into the `cf auth login` guidance
    if not isinstance(cfg, dict):
        raise ValueError(f"cf config is not an object: {type(cfg).__name__}")
    raw_token = cfg.get("oauth_token")
    if not isinstance(raw_token, str):
        raise ValueError("cf config oauth_token is not a string")
    token = raw_token.strip()
    expires = float("inf")
    raw_exp = cfg.get("expiration_time")
    if isinstance(raw_exp, str) and raw_exp:
        try:
            exp = datetime.fromisoformat(raw_exp.replace("Z", "+00:00"))
            expires = exp.timestamp()
        except ValueError:
            pass
    return token, expires


def _run_cf(*args: str) -> str:
    # shell=True: npm's Windows shims are .cmd files CreateProcess can't
    # launch by bare name; fixed argv, no user input, so this is safe
    out = subprocess.run(" ".join(("cf",) + args), shell=True,
                         capture_output=True, text=True, timeout=180)
    return out.stdout + out.stderr


def _cf_account_id() -> str:
    # every failure mode (missing CLI, no login, non-JSON or unexpected
    # output, timeout) returns "" so account_id() raises its guidance
    try:
        out = _run_cf("auth", "whoami")
        data = json.loads(out[out.find("{"):out.rfind("}") + 1])
        if not isinstance(data, dict):
            return ""
        accounts = data.get("accounts", [])
        if not isinstance(accounts, list):
            return ""
        for account in accounts:
            if isinstance(account, dict) and "id" in account:
                return str(account["id"])
    except (OSError, ValueError, subprocess.SubprocessError):
        pass
    return ""


def account_id() -> str:
    acc = _env_value("CLOUDFLARE_ACCOUNT_ID")
    if acc:
        return acc
    acc = _cf_account_id()
    if acc:
        return acc
    raise RuntimeError(
        "No CLOUDFLARE_ACCOUNT_ID found. Put it in the repo-root .env "
        "(shown by `cf auth whoami`) or set the environment variable."
    )


def load_token() -> str:
    """Bearer credential: explicit API token, else the cf CLI session."""
    token = _env_value("CLOUDFLARE_AUTH_TOKEN")
    if token:
        return token
    try:
        token, expires = _read_cf_session()
    except (OSError, ValueError, json.JSONDecodeError):
        raise RuntimeError(
            "No CLOUDFLARE_AUTH_TOKEN and no cf CLI login found. Run "
            "`cf auth login` (the session token is used and refreshed "
            "automatically), or create a Workers AI API token in the "
            "Cloudflare dash and set CLOUDFLARE_AUTH_TOKEN."
        ) from None
    if expires - time.time() < 120:  # refresh before it lapses mid-run
        try:
            _run_cf("auth", "whoami")  # any cf command refreshes it
            token, _ = _read_cf_session()
        except (OSError, ValueError, json.JSONDecodeError, subprocess.SubprocessError):
            pass  # let the API surface the auth failure instead
    return token


def clef(model: str = "clef", usage_sink=None):
    """A JevClient pointed at a hosted Clef model on Workers AI."""
    from engines.jev_client import JevClient

    if model not in MODELS:
        raise ValueError(f"unknown Clef model {model!r}; "
                         f"expected one of {sorted(MODELS)}")
    return JevClient(
        api_key=load_token(), model=model,
        base_url=f"{API_BASE}/accounts/{account_id()}/ai/run/{MODELS[model]}",
        usage_sink=usage_sink)
