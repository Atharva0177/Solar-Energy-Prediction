import os
import time
import json
import requests
from datetime import datetime

# ============================================================
# CONFIGURATION
# ============================================================

API_KEY = os.getenv("OPENROUTER_API_KEY", "YOUR_OPENROUTER_API_KEY")

# Change this to the model ID you are actually using through fcc-claude
MODEL = "oxalpha"

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

CHECK_INTERVAL = 10  # seconds


# ============================================================
# HELPERS
# ============================================================

def timestamp():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def print_rate_headers(response):
    print("\n--- Rate Limit Headers ---")

    headers = response.headers

    interesting = [
        "X-RateLimit-Limit",
        "X-RateLimit-Remaining",
        "X-RateLimit-Reset",
        "Retry-After",
    ]

    found = False

    for header in interesting:
        value = headers.get(header)

        if value is not None:
            print(f"{header}: {value}")
            found = True

    if not found:
        print("No X-RateLimit-* headers returned.")


# ============================================================
# CHECK OPENROUTER KEY / CREDIT LIMIT
# ============================================================

def check_key_limits():
    url = f"{OPENROUTER_BASE_URL}/key"

    headers = {
        "Authorization": f"Bearer {API_KEY}"
    }

    try:
        response = requests.get(
            url,
            headers=headers,
            timeout=15
        )

        print("\n" + "=" * 60)
        print(f"[{timestamp()}] API KEY STATUS")
        print("=" * 60)

        print(f"HTTP Status: {response.status_code}")

        if response.status_code != 200:
            print("Failed to retrieve key information.")
            print(response.text)
            return None

        data = response.json().get("data", {})

        print(f"Label:             {data.get('label')}")
        print(f"Credit Limit:      {data.get('limit')}")
        print(f"Credit Remaining:  {data.get('limit_remaining')}")
        print(f"Limit Reset:       {data.get('limit_reset')}")

        print(f"Usage Total:       {data.get('usage')}")
        print(f"Usage Daily:       {data.get('usage_daily')}")
        print(f"Usage Weekly:      {data.get('usage_weekly')}")
        print(f"Usage Monthly:     {data.get('usage_monthly')}")

        print(f"Free Tier:         {data.get('is_free_tier')}")

        return data

    except requests.RequestException as e:
        print(f"Key status request failed: {e}")
        return None


# ============================================================
# TEST CHAT REQUEST
# ============================================================

def test_model():
    url = f"{OPENROUTER_BASE_URL}/chat/completions"

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": MODEL,
        "messages": [
            {
                "role": "user",
                "content": "Reply with exactly: OK"
            }
        ],
        "max_tokens": 10,
        "stream": False
    }

    try:
        start = time.time()

        response = requests.post(
            url,
            headers=headers,
            json=payload,
            timeout=30
        )

        elapsed = time.time() - start

        print("\n" + "=" * 60)
        print(f"[{timestamp()}] MODEL TEST")
        print("=" * 60)

        print(f"Model:             {MODEL}")
        print(f"HTTP Status:       {response.status_code}")
        print(f"Response Time:     {elapsed:.2f}s")

        # ----------------------------------------------------
        # RATE LIMIT HEADERS
        # ----------------------------------------------------

        print_rate_headers(response)

        # ----------------------------------------------------
        # SUCCESS
        # ----------------------------------------------------

        if response.status_code == 200:
            print("\nSTATUS: OK")
            print("The request was accepted.")

            try:
                data = response.json()

                content = (
                    data
                    .get("choices", [{}])[0]
                    .get("message", {})
                    .get("content")
                )

                print(f"Response: {content}")

            except Exception:
                print("Could not parse successful response.")

            return {
                "success": True,
                "rate_limited": False
            }

        # ----------------------------------------------------
        # RATE LIMIT
        # ----------------------------------------------------

        if response.status_code == 429:
            print("\nSTATUS: RATE LIMITED")

            try:
                data = response.json()

                print("\nError:")
                print(json.dumps(data, indent=2))

                error = data.get("error", {})
                metadata = error.get("metadata", {})

                print("\n--- Rate Limit Diagnosis ---")

                print(f"Error Type:        {metadata.get('error_type')}")
                print(f"Provider Code:     {metadata.get('provider_code')}")

                if metadata.get("provider_code"):
                    print(
                        "\nThis strongly suggests the upstream provider "
                        "is rate limiting the request."
                    )
                else:
                    print(
                        "\nThis may be an OpenRouter platform-level "
                        "rate limit."
                    )

            except Exception:
                print("Could not parse 429 response.")
                print(response.text)

            return {
                "success": False,
                "rate_limited": True
            }

        # ----------------------------------------------------
        # CREDIT ERROR
        # ----------------------------------------------------

        if response.status_code == 402:
            print("\nSTATUS: CREDIT LIMIT / PAYMENT ERROR")

            try:
                print(
                    json.dumps(
                        response.json(),
                        indent=2
                    )
                )
            except Exception:
                print(response.text)

            return {
                "success": False,
                "credit_error": True
            }

        # ----------------------------------------------------
        # OTHER ERROR
        # ----------------------------------------------------

        print("\nSTATUS: ERROR")

        try:
            print(
                json.dumps(
                    response.json(),
                    indent=2
                )
            )
        except Exception:
            print(response.text)

        return {
            "success": False,
            "error": True
        }

    except requests.Timeout:
        print("\nSTATUS: TIMEOUT")
        print("The request timed out.")

        return {
            "success": False,
            "timeout": True
        }

    except requests.RequestException as e:
        print("\nSTATUS: CONNECTION ERROR")
        print(e)

        return {
            "success": False,
            "connection_error": True
        }


# ============================================================
# SINGLE CHECK
# ============================================================

def check_once():
    check_key_limits()
    test_model()


# ============================================================
# CONTINUOUS MONITOR
# ============================================================

def monitor():
    print("=" * 60)
    print("OpenRouter / fcc-claude Rate Limit Monitor")
    print("=" * 60)

    print(f"Model:           {MODEL}")
    print(f"Check interval:  {CHECK_INTERVAL}s")

    while True:
        try:
            check_once()

            print(
                f"\nNext check in {CHECK_INTERVAL} seconds..."
            )

            time.sleep(CHECK_INTERVAL)

        except KeyboardInterrupt:
            print("\n\nMonitor stopped.")
            break


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    if API_KEY == "YOUR_OPENROUTER_API_KEY":
        print("ERROR: Set your OpenRouter API key first.")
        print(
            "PowerShell example:"
        )
        print(
            '$env:OPENROUTER_API_KEY="sk-or-v1-..."'
        )
        raise SystemExit(1)

    check_once()

    # Uncomment this if you want continuous monitoring:
    # monitor()