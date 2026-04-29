"""
generate_token.py — Upstox API v2 Access Token Generator
==========================================================

Interactive script that walks you through the Upstox OAuth2 flow:
  1. Opens the authorization URL for you to login
  2. You paste the authorization code from the redirect
  3. Exchanges it for an access token
  4. Writes the token directly into your .env file

Usage:
    python generate_token.py
"""

import os
import sys
from urllib.parse import quote

import requests
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Load .env
# ---------------------------------------------------------------------------
ENV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
load_dotenv(ENV_PATH)

API_KEY = os.getenv("UPSTOX_API_KEY", "")
API_SECRET = os.getenv("UPSTOX_API_SECRET", "")
REDIRECT_URI = os.getenv("UPSTOX_REDIRECT_URI", "http://127.0.0.1")

TOKEN_URL = "https://api.upstox.com/v2/login/authorization/token"


def main() -> None:
    print("=" * 64)
    print("  Upstox API v2 — Access Token Generator")
    print("=" * 64)

    # ------------------------------------------------------------------
    # Validate credentials
    # ------------------------------------------------------------------
    if not API_KEY or API_KEY == "your_api_key_here":
        print("\n  ✘ UPSTOX_API_KEY is not set in .env")
        print("    Fill in your real API key and re-run.")
        sys.exit(1)

    if not API_SECRET or API_SECRET == "your_api_secret_here":
        print("\n  ✘ UPSTOX_API_SECRET is not set in .env")
        print("    Fill in your real API secret and re-run.")
        sys.exit(1)

    # ------------------------------------------------------------------
    # Step 1 — Build and print the login URL
    # ------------------------------------------------------------------
    encoded_redirect = quote(REDIRECT_URI, safe="")
    login_url = (
        f"https://api.upstox.com/v2/login/authorization/dialog"
        f"?response_type=code"
        f"&client_id={API_KEY}"
        f"&redirect_uri={encoded_redirect}"
    )

    print(f"\n  Step 1: Open this URL in your browser and login:\n")
    print(f"  {login_url}\n")

    # ------------------------------------------------------------------
    # Step 2 — Wait for user to paste the authorization code
    # ------------------------------------------------------------------
    print("  After login, you will be redirected to a URL like:")
    print(f"  {REDIRECT_URI}?code=XXXXXX\n")
    print("  Copy the 'code' value from that URL and paste it below.\n")

    auth_code = input("  Enter authorization code: ").strip()

    if not auth_code:
        print("\n  ✘ No code entered. Aborting.")
        sys.exit(1)

    # ------------------------------------------------------------------
    # Step 3 — Exchange code for access token
    # ------------------------------------------------------------------
    print("\n  Exchanging code for access token...")

    try:
        response = requests.post(
            TOKEN_URL,
            data={
                "code": auth_code,
                "client_id": API_KEY,
                "client_secret": API_SECRET,
                "redirect_uri": REDIRECT_URI,
                "grant_type": "authorization_code",
            },
            headers={
                "Content-Type": "application/x-www-form-urlencoded",
                "Accept": "application/json",
            },
        )

        response_json = response.json()

        if response.status_code != 200 or "access_token" not in response_json:
            print(f"\n  ✘ Token exchange failed (HTTP {response.status_code}):")
            print(f"  {response_json}")
            sys.exit(1)

        access_token = response_json["access_token"]
        print(f"\n  ✔ Access token received: {access_token[:20]}...{access_token[-10:]}")

    except requests.RequestException as exc:
        print(f"\n  ✘ Network error during token exchange: {exc}")
        sys.exit(1)
    except ValueError as exc:
        print(f"\n  ✘ Could not parse response as JSON: {exc}")
        print(f"  Raw response: {response.text}")
        sys.exit(1)

    # ------------------------------------------------------------------
    # Step 4 — Write token to .env file
    # ------------------------------------------------------------------
    print(f"\n  Writing token to {ENV_PATH}...")

    try:
        if not os.path.exists(ENV_PATH):
            print(f"\n  ✘ .env file not found at {ENV_PATH}")
            print(f"  Manually set: UPSTOX_ACCESS_TOKEN={access_token}")
            sys.exit(1)

        with open(ENV_PATH, "r", encoding="utf-8") as f:
            lines = f.readlines()

        token_line_found = False
        new_lines = []
        for line in lines:
            if line.startswith("UPSTOX_ACCESS_TOKEN="):
                new_lines.append(f"UPSTOX_ACCESS_TOKEN={access_token}\n")
                token_line_found = True
            else:
                new_lines.append(line)

        # If the key didn't exist, append it
        if not token_line_found:
            new_lines.append(f"UPSTOX_ACCESS_TOKEN={access_token}\n")

        with open(ENV_PATH, "w", encoding="utf-8") as f:
            f.writelines(new_lines)

        print("\n  ✔ Token saved to .env successfully. Bot is ready to run.")

    except OSError as exc:
        print(f"\n  ✘ Failed to write .env file: {exc}")
        print(f"  Manually set: UPSTOX_ACCESS_TOKEN={access_token}")
        sys.exit(1)

    print("\n" + "=" * 64)
    print("  Done. You can now run:  python main.py")
    print("=" * 64)


if __name__ == "__main__":
    main()
