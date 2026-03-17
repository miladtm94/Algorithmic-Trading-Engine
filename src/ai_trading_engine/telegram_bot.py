from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request


class TelegramError(RuntimeError):
    pass


def send_message(token: str, chat_id: str, text: str) -> dict:
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = urllib.parse.urlencode(
        {
            "chat_id": chat_id,
            "text": text,
        }
    ).encode("utf-8")

    req = urllib.request.Request(
        url=url,
        data=payload,
        method="POST",
        headers={"Content-Type": "application/x-www-form-urlencoded"},
    )
    try:
        with urllib.request.urlopen(req, timeout=20) as response:
            body = response.read().decode("utf-8")
            data = json.loads(body)
    except urllib.error.URLError as exc:
        raise TelegramError(f"Telegram API request failed: {exc}") from exc

    if not data.get("ok"):
        raise TelegramError(f"Telegram API error: {data}")
    return data
