"""Local OAuth 2.0 + OpenID Connect (OIDC) Identity Provider for RAKSHA AI.

Implements:
- OIDC Discovery:   GET  /.well-known/openid-configuration
- RSA JWKS:         GET  /.well-known/jwks.json
- Auth Endpoint:    GET/POST /oauth2/authorize (Authorization Code + PKCE S256 + nonce)
- Token Endpoint:   POST /oauth2/token (client_secret_basic/post, PKCE verification, RS256 id_token)
- UserInfo:         GET  /oauth2/userinfo (Bearer access_token)
- OIDC End Session: GET  /oauth2/logout

Users are stored in data/processed/users.db with PBKDF2-HMAC-SHA256 salted password hashes.
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import html
import json
import os
import secrets
import sqlite3
import time
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlencode, urlparse

import uvicorn
from joserfc import jwt
from joserfc.jwk import KeySet, RSAKey
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import HTMLResponse, JSONResponse, RedirectResponse
from starlette.routing import Route

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "processed"
USER_DB = DATA_DIR / "users.db"
KEY_FILE = DATA_DIR / "oidc_rsa_key.json"

ISSUER = os.environ.get("RAKSHA_OIDC_ISSUER", "http://localhost:8502")
CLIENT_ID = os.environ.get("RAKSHA_OIDC_CLIENT_ID", "raksha-oidc-client")
CLIENT_SECRET = os.environ.get("RAKSHA_OIDC_CLIENT_SECRET", "raksha-oidc-secret-2026")
ALLOWED_REDIRECT_URIS = {
    "http://localhost:8501/oauth2callback",
    "http://127.0.0.1:8501/oauth2callback",
}
KID = "raksha-oidc-rsa-1"

_AUTH_CODES: dict[str, dict] = {}
_ACCESS_TOKENS: dict[str, dict] = {}

GOOGLE_G_SVG = (
    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 48 48" width="22" height="22">'
    '<path fill="#FFC107" d="M43.611,20.083H42V20H24v8h11.303c-1.649,4.657-6.08,8-11.303,8c-6.627,0-12-5.373-12-12'
    'c0-6.627,5.373-12,12-12c3.059,0,5.842,1.154,7.961,3.039l5.657-5.657C34.046,6.053,29.268,4,24,4'
    'C12.955,4,4,12.955,4,24c0,11.045,8.955,20,20,20c11.045,0,20-8.955,20-20C44,22.659,43.862,21.35,43.611,20.083z"/>'
    '<path fill="#FF3D00" d="M6.306,14.691l6.571,4.819C14.655,15.108,18.961,12,24,12c3.059,0,5.842,1.154,7.961,3.039'
    'l5.657-5.657C34.046,6.053,29.268,4,24,4C16.318,4,9.656,8.337,6.306,14.691z"/>'
    '<path fill="#4CAF50" d="M24,44c5.166,0,9.86-1.977,13.409-5.192l-6.19-5.238C29.211,35.091,26.715,36,24,36'
    'c-5.202,0-9.619-3.317-11.283-7.946l-6.522,5.025C9.505,39.556,16.227,44,24,44z"/>'
    '<path fill="#1976D2" d="M43.611,20.083H42V20H24v8h11.303c-0.792,2.237-2.231,4.166-4.087,5.571'
    'c0.001-0.001,0.002-0.001,0.003-0.002l6.19,5.238C36.971,39.205,44,34,44,24C44,22.659,43.862,21.35,43.611,20.083z"/>'
    "</svg>"
)


def _load_or_create_rsa_key() -> RSAKey:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if KEY_FILE.exists():
        data = json.loads(KEY_FILE.read_text(encoding="utf-8"))
        return RSAKey.import_key(data)
    key = RSAKey.generate_key(2048, {"kid": KID, "use": "sig", "alg": "RS256"})
    KEY_FILE.write_text(json.dumps(key.as_dict(private=True), indent=2), encoding="utf-8")
    return key


RSA_KEY = _load_or_create_rsa_key()
JWKS_PUBLIC = KeySet([RSA_KEY]).as_dict(private=False)


# ---------------------------------------------------------------------------
# User Database Helpers (SQLite + PBKDF2-HMAC-SHA256)
# ---------------------------------------------------------------------------
def _init_user_db() -> None:
    USER_DB.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(USER_DB) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                email TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                password_hash TEXT NOT NULL,
                salt TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )


def _hash_password(password: str, salt: str | None = None) -> tuple[str, str]:
    s = salt or secrets.token_hex(16)
    h = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), s.encode("utf-8"), 200_000).hex()
    return h, s


def _get_user(email: str) -> dict | None:
    _init_user_db()
    with sqlite3.connect(USER_DB) as conn:
        row = conn.execute(
            "SELECT email, name, password_hash, salt FROM users WHERE email = ?",
            (email.strip().lower(),),
        ).fetchone()
    if not row:
        return None
    return {"email": row[0], "name": row[1], "password_hash": row[2], "salt": row[3]}


def _create_user(email: str, name: str, password: str) -> dict:
    _init_user_db()
    em = email.strip().lower()
    pw_hash, salt = _hash_password(password)
    now = datetime.now(UTC).isoformat()
    with sqlite3.connect(USER_DB) as conn:
        conn.execute(
            "INSERT INTO users (email, name, password_hash, salt, created_at) VALUES (?, ?, ?, ?, ?)",
            (em, name.strip(), pw_hash, salt, now),
        )
    return {"email": em, "name": name.strip()}


def _verify_user(email: str, password: str) -> dict | None:
    u = _get_user(email)
    if not u:
        return None
    pw_hash, _ = _hash_password(password, u["salt"])
    if hmac.compare_digest(pw_hash, u["password_hash"]):
        return {"email": u["email"], "name": u["name"]}
    return None


def _sub_for_email(email: str) -> str:
    return hashlib.sha256(email.strip().lower().encode("utf-8")).hexdigest()[:24]


def _at_hash(access_token: str) -> str:
    digest = hashlib.sha256(access_token.encode("ascii")).digest()
    return base64.urlsafe_b64encode(digest[:16]).decode("ascii").rstrip("=")


def _pkce_s256(verifier: str) -> str:
    digest = hashlib.sha256(verifier.encode("ascii")).digest()
    return base64.urlsafe_b64encode(digest).decode("ascii").rstrip("=")


# ---------------------------------------------------------------------------
# OIDC HTML Sign-In Page
# ---------------------------------------------------------------------------
def _render_authorize_html(
    params: dict[str, str],
    error: str = "",
    active_tab: str = "login",
    prefill_email: str = "",
) -> str:
    hidden = "\n".join(
        f'<input type="hidden" name="{html.escape(k)}" value="{html.escape(v)}">'
        for k, v in params.items()
    )
    err_html = f'<div class="err">{html.escape(error)}</div>' if error else ""
    scope_txt = html.escape(params.get("scope", "openid email profile"))
    pkce_txt = html.escape(params.get("code_challenge_method", "none"))
    cid_txt = html.escape(params.get("client_id", ""))
    em_val = html.escape(prefill_email)
    login_display = "block" if active_tab == "login" else "none"
    reg_display = "block" if active_tab == "register" else "none"
    login_cls = "tab active" if active_tab == "login" else "tab"
    reg_cls = "tab active" if active_tab == "register" else "tab"

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Sign in — OAuth 2.0 / OpenID Connect</title>
  <style>
    * {{ box-sizing: border-box; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }}
    body {{
      margin: 0; min-height: 100vh; background: #F8FAFC; color: #0F172A;
      display: flex; align-items: center; justify-content: center; padding: 24px;
    }}
    .card {{
      width: 100%; max-width: 430px; background: #FFFFFF; border: 1px solid #CBD5E1;
      border-radius: 14px; padding: 28px 28px 24px; box-shadow: 0 8px 28px rgba(15, 23, 42, 0.06);
    }}
    .hdr {{ display: flex; align-items: center; gap: 10px; margin-bottom: 6px; }}
    .hdr h1 {{ font-size: 1.28rem; margin: 0; font-weight: 700; }}
    .sub {{ font-size: 0.88rem; color: #475569; margin: 0 0 16px; }}
    .oidc-meta {{
      background: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 8px;
      padding: 8px 12px; font-size: 0.76rem; color: #475569; margin-bottom: 16px; line-height: 1.45;
    }}
    .oidc-meta code {{ color: #0F172A; font-weight: 600; }}
    .tabs {{ display: grid; grid-template-columns: 1fr 1fr; gap: 6px; background: #F1F5F9; padding: 4px; border-radius: 9px; margin-bottom: 16px; }}
    .tab {{
      border: none; background: transparent; padding: 8px; border-radius: 6px;
      font-weight: 600; font-size: 0.85rem; color: #475569; cursor: pointer;
    }}
    .tab.active {{ background: #FFFFFF; color: #0F172A; box-shadow: 0 1px 3px rgba(15,23,42,0.08); }}
    label {{ display: block; font-size: 0.82rem; font-weight: 600; color: #334155; margin: 10px 0 4px; }}
    input[type=email], input[type=password], input[type=text] {{
      width: 100%; padding: 10px 12px; border: 1px solid #CBD5E1; border-radius: 8px;
      font-size: 0.92rem; outline: none;
    }}
    input:focus {{ border-color: #2563EB; box-shadow: 0 0 0 3px rgba(37,99,235,0.12); }}
    .btn {{
      width: 100%; margin-top: 16px; padding: 11px; border: none; border-radius: 8px;
      background: #2563EB; color: #FFFFFF; font-size: 0.94rem; font-weight: 600; cursor: pointer;
    }}
    .btn:hover {{ background: #1D4ED8; }}
    .err {{
      background: #FEF2F2; border: 1px solid #FECACA; color: #991B1B;
      padding: 9px 12px; border-radius: 8px; font-size: 0.84rem; margin-bottom: 12px;
    }}
    .back {{ display: inline-block; margin-top: 14px; font-size: 0.82rem; color: #475569; text-decoration: none; }}
    .back:hover {{ color: #0F172A; text-decoration: underline; }}
  </style>
</head>
<body>
  <div class="card">
    <div class="hdr">{GOOGLE_G_SVG}<h1>Sign in with OIDC</h1></div>
    <p class="sub">Authorize <b>RAKSHA AI</b> via OAuth 2.0 &amp; OpenID Connect</p>

    <div class="oidc-meta">
      Client: <code>{cid_txt}</code> &bull; Scopes: <code>{scope_txt}</code> &bull; PKCE: <code>{pkce_txt}</code>
    </div>

    {err_html}

    <div class="tabs">
      <button type="button" id="tab-login" class="{login_cls}" onclick="showTab('login')">Sign In</button>
      <button type="button" id="tab-register" class="{reg_cls}" onclick="showTab('register')">Create Account</button>
    </div>

    <form id="form-login" method="post" action="/oauth2/authorize" style="display:{login_display};">
      {hidden}
      <input type="hidden" name="action" value="login" />
      <label for="l-email">Email</label>
      <input id="l-email" name="email" type="email" value="{em_val}" placeholder="you@gmail.com" required />
      <label for="l-pass">Password</label>
      <input id="l-pass" name="password" type="password" placeholder="Enter your password" required />
      <button class="btn" type="submit">Continue &amp; Authorize</button>
    </form>

    <form id="form-register" method="post" action="/oauth2/authorize" style="display:{reg_display};">
      {hidden}
      <input type="hidden" name="action" value="register" />
      <label for="r-name">Full Name</label>
      <input id="r-name" name="name" type="text" placeholder="Your full name" required />
      <label for="r-email">Email</label>
      <input id="r-email" name="email" type="email" value="{em_val}" placeholder="you@gmail.com" required />
      <label for="r-pass1">Password (min 6 characters)</label>
      <input id="r-pass1" name="password" type="password" placeholder="Create password" minlength="6" required />
      <label for="r-pass2">Confirm Password</label>
      <input id="r-pass2" name="confirm_password" type="password" placeholder="Re-enter password" minlength="6" required />
      <button class="btn" type="submit">Create Account &amp; Authorize</button>
    </form>

    <a class="back" href="http://localhost:8501/">Cancel and return to RAKSHA AI</a>
  </div>
  <script>
    function showTab(mode) {{
      document.getElementById('form-login').style.display = mode === 'login' ? 'block' : 'none';
      document.getElementById('form-register').style.display = mode === 'register' ? 'block' : 'none';
      document.getElementById('tab-login').className = mode === 'login' ? 'tab active' : 'tab';
      document.getElementById('tab-register').className = mode === 'register' ? 'tab active' : 'tab';
    }}
  </script>
</body>
</html>"""


# ---------------------------------------------------------------------------
# Endpoint Handlers
# ---------------------------------------------------------------------------
async def discovery(request: Request) -> JSONResponse:
    base = str(request.base_url).rstrip("/")
    return JSONResponse(
        {
            "issuer": ISSUER,
            "authorization_endpoint": f"{base}/oauth2/authorize",
            "token_endpoint": f"{base}/oauth2/token",
            "userinfo_endpoint": f"{base}/oauth2/userinfo",
            "jwks_uri": f"{base}/.well-known/jwks.json",
            "end_session_endpoint": f"{base}/oauth2/logout",
            "response_types_supported": ["code"],
            "subject_types_supported": ["public"],
            "id_token_signing_alg_values_supported": ["RS256"],
            "scopes_supported": ["openid", "email", "profile"],
            "token_endpoint_auth_methods_supported": ["client_secret_basic", "client_secret_post"],
            "code_challenge_methods_supported": ["S256", "plain"],
            "claims_supported": ["sub", "iss", "aud", "exp", "iat", "nonce", "email", "email_verified", "name"],
        }
    )


async def jwks(request: Request) -> JSONResponse:
    return JSONResponse(JWKS_PUBLIC)


async def health(request: Request) -> JSONResponse:
    return JSONResponse({"status": "ok", "issuer": ISSUER})


def _extract_oidc_params(source) -> dict[str, str]:
    keys = ["client_id", "redirect_uri", "response_type", "scope", "state", "nonce", "code_challenge", "code_challenge_method"]
    return {k: str(source.get(k, "")) for k in keys if source.get(k) is not None}


def _validate_auth_request(params: dict[str, str]) -> str | None:
    if params.get("client_id") != CLIENT_ID:
        return "Invalid client_id."
    if params.get("redirect_uri") not in ALLOWED_REDIRECT_URIS:
        return "Unregistered redirect_uri."
    if params.get("response_type") != "code":
        return "Unsupported response_type (expected code)."
    return None


async def authorize_get(request: Request) -> HTMLResponse:
    params = _extract_oidc_params(request.query_params)
    err = _validate_auth_request(params)
    if err:
        return HTMLResponse(f"<h3>OAuth 2.0 Error</h3><p>{html.escape(err)}</p>", status_code=400)
    return HTMLResponse(_render_authorize_html(params))


async def authorize_post(request: Request) -> HTMLResponse | RedirectResponse:
    form = await request.form()
    params = _extract_oidc_params(form)
    err = _validate_auth_request(params)
    if err:
        return HTMLResponse(f"<h3>OAuth 2.0 Error</h3><p>{html.escape(err)}</p>", status_code=400)

    action = str(form.get("action", "login"))
    email = str(form.get("email", "")).strip().lower()
    password = str(form.get("password", ""))

    if action == "register":
        name = str(form.get("name", "")).strip()
        confirm = str(form.get("confirm_password", ""))
        parts = email.split("@")
        if not name:
            return HTMLResponse(_render_authorize_html(params, "Please enter your full name.", "register", email), status_code=400)
        if len(parts) != 2 or not parts[0] or "." not in parts[1]:
            return HTMLResponse(_render_authorize_html(params, "Enter a valid email address.", "register", email), status_code=400)
        if len(password) < 6:
            return HTMLResponse(_render_authorize_html(params, "Password must be at least 6 characters.", "register", email), status_code=400)
        if password != confirm:
            return HTMLResponse(_render_authorize_html(params, "Passwords do not match.", "register", email), status_code=400)
        if _get_user(email) is not None:
            return HTMLResponse(_render_authorize_html(params, "Email already exists. Please sign in.", "login", email), status_code=400)
        user = _create_user(email, name, password)
    else:
        existing = _get_user(email)
        if existing is None:
            return HTMLResponse(
                _render_authorize_html(params, "No account found for that email. Create an account below.", "register", email),
                status_code=401,
            )
        user = _verify_user(email, password)
        if user is None:
            return HTMLResponse(_render_authorize_html(params, "Incorrect password. Please try again.", "login", email), status_code=401)

    code = secrets.token_urlsafe(32)
    now = int(time.time())
    _AUTH_CODES[code] = {
        "client_id": params["client_id"],
        "redirect_uri": params["redirect_uri"],
        "scope": params.get("scope", "openid email profile"),
        "nonce": params.get("nonce", ""),
        "code_challenge": params.get("code_challenge", ""),
        "code_challenge_method": params.get("code_challenge_method", ""),
        "user": user,
        "auth_time": now,
        "expires_at": now + 300,
    }
    qs = {"code": code}
    if params.get("state"):
        qs["state"] = params["state"]
    target = f"{params['redirect_uri']}?{urlencode(qs)}"
    return RedirectResponse(target, status_code=302)


async def token_endpoint(request: Request) -> JSONResponse:
    form = await request.form()

    # Support client_secret_basic and client_secret_post
    cid = str(form.get("client_id", ""))
    csec = str(form.get("client_secret", ""))
    auth_hdr = request.headers.get("authorization", "")
    if auth_hdr.lower().startswith("basic "):
        try:
            decoded = base64.b64decode(auth_hdr[6:].strip()).decode("utf-8")
            cid, csec = decoded.split(":", 1)
        except Exception:
            return JSONResponse({"error": "invalid_client"}, status_code=401)

    if cid != CLIENT_ID or not hmac.compare_digest(csec, CLIENT_SECRET):
        return JSONResponse({"error": "invalid_client"}, status_code=401)

    if form.get("grant_type") != "authorization_code":
        return JSONResponse({"error": "unsupported_grant_type"}, status_code=400)

    code = str(form.get("code", ""))
    entry = _AUTH_CODES.pop(code, None)
    now = int(time.time())
    if not entry or entry["expires_at"] < now:
        return JSONResponse({"error": "invalid_grant", "error_description": "Invalid or expired code"}, status_code=400)
    if entry["client_id"] != cid or entry["redirect_uri"] != str(form.get("redirect_uri", "")):
        return JSONResponse({"error": "invalid_grant", "error_description": "redirect_uri or client_id mismatch"}, status_code=400)

    # PKCE verification
    if entry["code_challenge"]:
        verifier = str(form.get("code_verifier", ""))
        method = entry["code_challenge_method"] or "plain"
        if not verifier:
            return JSONResponse({"error": "invalid_grant", "error_description": "Missing code_verifier"}, status_code=400)
        if method == "S256":
            expected = _pkce_s256(verifier)
        elif method == "plain":
            expected = verifier
        else:
            return JSONResponse({"error": "invalid_grant", "error_description": "Unsupported PKCE method"}, status_code=400)
        if not hmac.compare_digest(expected, entry["code_challenge"]):
            return JSONResponse({"error": "invalid_grant", "error_description": "PKCE verification failed"}, status_code=400)

    user = entry["user"]
    sub = _sub_for_email(user["email"])
    access_token = secrets.token_urlsafe(32)
    _ACCESS_TOKENS[access_token] = {
        "sub": sub,
        "email": user["email"],
        "email_verified": True,
        "name": user["name"],
        "scope": entry["scope"],
        "client_id": cid,
        "expires_at": now + 3600,
    }

    claims = {
        "iss": ISSUER,
        "sub": sub,
        "aud": cid,
        "iat": now,
        "exp": now + 3600,
        "auth_time": entry["auth_time"],
        "at_hash": _at_hash(access_token),
        "email": user["email"],
        "email_verified": True,
        "name": user["name"],
    }
    if entry["nonce"]:
        claims["nonce"] = entry["nonce"]

    id_token = jwt.encode({"alg": "RS256", "kid": KID, "typ": "JWT"}, claims, RSA_KEY)
    return JSONResponse(
        {
            "access_token": access_token,
            "token_type": "Bearer",
            "expires_in": 3600,
            "id_token": id_token,
            "scope": entry["scope"],
        }
    )


async def userinfo_endpoint(request: Request) -> JSONResponse:
    auth_hdr = request.headers.get("authorization", "")
    if not auth_hdr.lower().startswith("bearer "):
        return JSONResponse({"error": "invalid_token"}, status_code=401)
    tok = auth_hdr[7:].strip()
    info = _ACCESS_TOKENS.get(tok)
    if not info or info["expires_at"] < int(time.time()):
        return JSONResponse({"error": "invalid_token"}, status_code=401)
    return JSONResponse(
        {
            "sub": info["sub"],
            "email": info["email"],
            "email_verified": info["email_verified"],
            "name": info["name"],
        }
    )


async def logout_endpoint(request: Request) -> RedirectResponse:
    post_uri = request.query_params.get("post_logout_redirect_uri", "http://localhost:8501/")
    parsed = urlparse(post_uri)
    if parsed.scheme and parsed.netloc:
        target = f"{parsed.scheme}://{parsed.netloc}/"
    else:
        target = "http://localhost:8501/"
    return RedirectResponse(target, status_code=302)


app = Starlette(
    routes=[
        Route("/.well-known/openid-configuration", discovery, methods=["GET"]),
        Route("/.well-known/jwks.json", jwks, methods=["GET"]),
        Route("/health", health, methods=["GET"]),
        Route("/oauth2/authorize", authorize_get, methods=["GET"]),
        Route("/oauth2/authorize", authorize_post, methods=["POST"]),
        Route("/oauth2/token", token_endpoint, methods=["POST"]),
        Route("/oauth2/userinfo", userinfo_endpoint, methods=["GET"]),
        Route("/oauth2/logout", logout_endpoint, methods=["GET"]),
    ]
)


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8502, log_level="warning")
