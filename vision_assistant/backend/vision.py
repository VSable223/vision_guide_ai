import requests

LLAMA_URL = "http://127.0.0.1:8080/v1/chat/completions"

TOKEN_PROFILES = {
    "live": 20,
    "qa": 40,
    "image_summary": 160,
    "video_summary": 320,
}

LIVE_SYSTEM_PROMPT = (
    "You are a real-time vision narrator.\n"
    "Rules:\n"
    "- Describe ONLY what changed since the previous frame.\n"
    "- Use AT MOST 6 words.\n"
    "- Focus ONLY on actions or presence.\n"
    "- No explanations.\n"
    "- No full sentences.\n"
    "- If nothing changed, reply exactly: no change.\n"
)

# ---------- IMAGE / LIVE / QA ----------
def query_vlm(
    image_b64,
    system_prompt,
    user_text="",
    profile="qa"
):
    max_tokens = TOKEN_PROFILES.get(profile, 80)
    temperature = 0.1 if profile == "live" else 0.2

    user_content = []
    if user_text:
        user_content.append({"type": "text", "text": user_text})
    if image_b64:
        user_content.append(
            {"type": "image_url", "image_url": {"url": image_b64}}
        )

    payload = {
        "max_tokens": max_tokens,
        "temperature": temperature,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content}
        ]
    }

    r = requests.post(LLAMA_URL, json=payload, timeout=60)
    r.raise_for_status()
    text = r.json()["choices"][0]["message"]["content"].strip()


    return text


# ---------- TEXT-ONLY (VIDEO SUMMARY FINAL STEP) ----------
def query_text_only(
    system_prompt,
    user_text,
    profile="video_summary"
):
    max_tokens = TOKEN_PROFILES.get(profile, 320)

    payload = {
        "max_tokens": max_tokens,
        "temperature": 0.3,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_text}
        ]
    }

    r = requests.post(LLAMA_URL, json=payload, timeout=60)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"].strip()
