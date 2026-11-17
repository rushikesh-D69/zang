"""Provider-neutral LLM + speech-to-text clients (OpenAI-compatible HTTP APIs, stdlib only).

Providers are tried in order; the first that answers wins (dev: Groq free tier, fallback: Mistral;
production: point ZANG_LLM_PROVIDERS at "openai"). Keys come from the environment or from
config/secrets.env (local file, never sent to the browser):

    GROQ_API_KEY=...      MISTRAL_API_KEY=...      OPENAI_API_KEY=...
    ZANG_LLM_PROVIDERS=groq,mistral                (order)
    ZANG_GROQ_MODEL=qwen/qwen3.8-27b        ZANG_MISTRAL_MODEL=mistral-small-latest

STT: Groq Whisper (whisper-large-v3-turbo) when a key is set, else local faster-whisper (models/whisper).
"""
import io
import json
import os
import time
import urllib.error
import urllib.request
import uuid
import wave

import numpy as np

from ..paths import ROOT

SECRETS = os.path.join(ROOT, "config", "secrets.env")

PROVIDERS = {
    "groq": {"url": "https://api.groq.com/openai/v1", "key": "GROQ_API_KEY", "model_env": "ZANG_GROQ_MODEL",
             "model": "qwen/qwen3.8-27b", "small_model": "qwen/qwen3.8-27b", "stt_model": "whisper-large-v3-turbo",
             # Groq retires models often (llama-3.3-70b-versatile went away in 2026): tried in order, first one wins
             "fallback_models": ["openai/gpt-oss-120b", "openai/gpt-oss-20b", "llama-3.3-70b-versatile"]},
    "mistral": {"url": "https://api.mistral.ai/v1", "key": "MISTRAL_API_KEY", "model_env": "ZANG_MISTRAL_MODEL",
                "model": "mistral-small-latest", "small_model": "ministral-8b-latest"},
    "openai": {"url": "https://api.openai.com/v1", "key": "OPENAI_API_KEY", "model_env": "ZANG_OPENAI_MODEL",
               "model": "gpt-4o-mini", "small_model": "gpt-4o-mini", "stt_model": "whisper-1"},
}


def load_secrets():
    """config/secrets.env -> os.environ (environment variables win)."""
    if os.path.exists(SECRETS):
        for line in open(SECRETS, encoding="utf-8"):
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"'))


def save_secret(name, value):
    """Store a key in config/secrets.env (mode 600)."""
    lines = []
    if os.path.exists(SECRETS):
        lines = [ln for ln in open(SECRETS, encoding="utf-8").read().splitlines() if not ln.startswith(name + "=")]
    lines.append(f"{name}={value.strip()}")
    with open(SECRETS, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    os.chmod(SECRETS, 0o600)
    os.environ[name] = value.strip()


def order():
    load_secrets()
    names = [n.strip() for n in os.environ.get("ZANG_LLM_PROVIDERS", "groq,mistral").split(",") if n.strip() in PROVIDERS]
    return [n for n in names if os.environ.get(PROVIDERS[n]["key"])]


def status():
    load_secrets()
    return {n: {"key_set": bool(os.environ.get(p["key"])), "model": os.environ.get(p["model_env"], p["model"])}
            for n, p in PROVIDERS.items()} | {"order": order()}


def _req(provider, path, body=None, data=None, headers=None, timeout=60):
    p = PROVIDERS[provider]
    h = {"Authorization": f"Bearer {os.environ[p['key']]}", "User-Agent": "zang-ai/0.1"}
    if body is not None:
        data = json.dumps(body).encode()
        h["Content-Type"] = "application/json"
    h.update(headers or {})
    return urllib.request.urlopen(urllib.request.Request(p["url"] + path, data, h), timeout=timeout)


class LLMError(RuntimeError):
    pass


def _model_extra(model):
    """Reasoning models think before answering; for spoken replies we want the first word fast."""
    if model.startswith("qwen/qwen3"):
        return {"reasoning_effort": "none"}       # 0.19 s to first word on Groq (gpt-oss low: ~0.5 s)
    if model.startswith("openai/gpt-oss"):
        return {"reasoning_effort": "low"}
    return {}


_working = {}  # provider -> model that answered last time (skip retired ones)


def _models(name, small):
    p = PROVIDERS[name]
    first = p.get("small_model", p["model"]) if small else os.environ.get(p["model_env"], p["model"])
    out = [first] + [m for m in p.get("fallback_models", []) if m != first]
    w = _working.get(name)
    return ([w] + [m for m in out if m != w]) if w in out else out


def stream_chat(messages, cancel=None, temperature=0.7, max_tokens=600, small=False):  # rhymes / stories need room
    """Yield text deltas from the first provider that answers. Yields ("provider", name) first.

    cancel: threading.Event; when set, the stream is closed (barge-in). small: use the provider's cheap fast
    model (background jobs such as memory extraction). If every model is rate-limited (free tier: tokens per
    minute, resets within seconds), wait 4 s and try once more instead of failing the turn."""
    for attempt in range(2):
        try:
            yield from _stream_chat(messages, cancel, temperature, max_tokens, small)
            return
        except LLMError as e:
            if attempt or "HTTP 429" not in str(e):
                raise
            if cancel is not None:
                if cancel.wait(4.0):
                    return
            else:
                time.sleep(4.0)


def _stream_chat(messages, cancel, temperature, max_tokens, small):
    errors = []
    for name in order():
        r = None
        for model in _models(name, small):
            try:
                r = _req(name, "/chat/completions", {"model": model, "messages": messages, "stream": True,
                                                     "temperature": temperature, "max_tokens": max_tokens,
                                                     **_model_extra(model)}, timeout=30)
                _working[name] = model
                break
            except urllib.error.HTTPError as e:
                body = e.read()[:200]
                errors.append(f"{name}:{model}: HTTP {e.code} {body!r}")
                if (e.code in (400, 404) and b"model" in body) or e.code == 429:
                    continue  # retired / not on this key / rate-limited (limits are per model): try the next model
                break
            except Exception as e:  # network
                errors.append(f"{name}: {e}")
                break
        if r is None:
            continue
        yield ("provider", f"{name}:{model}")
        try:
            for raw in r:
                if cancel is not None and cancel.is_set():
                    r.close()
                    return
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    return
                try:
                    delta = json.loads(payload)["choices"][0].get("delta", {}).get("content")
                except (ValueError, KeyError, IndexError):
                    continue
                if delta:
                    yield ("text", delta)
            return
        except Exception as e:
            errors.append(f"{name} (mid-stream): {e}")
            return
    raise LLMError("no LLM provider answered: " + ("; ".join(errors) or "no API key set (GROQ_API_KEY / MISTRAL_API_KEY)"))


def _wav_bytes(audio16k):
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes((np.clip(audio16k, -1, 1) * 32767).astype(np.int16).tobytes())
    return buf.getvalue()


def transcribe(audio16k, language=None, prompt="Zang is a robot."):
    """-> (text, engine). Cloud Whisper if a key is set (fast), else local faster-whisper."""
    load_secrets()
    for name in [n for n in order() if "stt_model" in PROVIDERS[n]]:
        try:
            boundary = uuid.uuid4().hex
            fields = {"model": PROVIDERS[name]["stt_model"], "response_format": "json", "prompt": prompt}
            if language:
                fields["language"] = language
            body = b""
            for k, v in fields.items():
                body += f"--{boundary}\r\nContent-Disposition: form-data; name=\"{k}\"\r\n\r\n{v}\r\n".encode()
            body += (f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"turn.wav\"\r\n"
                     f"Content-Type: audio/wav\r\n\r\n").encode() + _wav_bytes(audio16k) + f"\r\n--{boundary}--\r\n".encode()
            r = _req(name, "/audio/transcriptions", data=body,
                     headers={"Content-Type": f"multipart/form-data; boundary={boundary}"}, timeout=30)
            return json.load(r).get("text", "").strip(), f"{name}-whisper"
        except Exception as e:
            print(f"cloud STT {name} failed: {e}")
    return local_transcribe(audio16k, "base", language, prompt), "local-whisper-base"


_local = {}


def local_transcribe(audio16k, size="base", language=None, prompt=None):
    from faster_whisper import WhisperModel
    if size not in _local:
        _local[size] = WhisperModel(size, device="cpu", compute_type="int8", cpu_threads=4,
                                    download_root=os.path.join(ROOT, "models", "whisper"))
    segs, _ = _local[size].transcribe(np.asarray(audio16k, np.float32), beam_size=1, language=language,
                                      initial_prompt=prompt, without_timestamps=True, condition_on_previous_text=False)
    return " ".join(s.text for s in segs).strip()


def ping():
    """Cheap check of each configured provider (lists models)."""
    out = {}
    for name in order():
        t0 = time.time()
        try:
            ids = [m["id"] for m in json.load(_req(name, "/models", timeout=10)).get("data", [])]
            out[name] = {"ok": True, "ms": round(1000 * (time.time() - t0)), "models": ids[:40]}
        except Exception as e:
            out[name] = {"ok": False, "error": str(e)}
    return out
