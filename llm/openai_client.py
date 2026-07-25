"""
OpenAI-uyumlu ince istemci — OpenRouter, vLLM, ya da herhangi bir
OpenAI-uyumlu endpoint için TEK sarmalayıcı.

Sağlayıcı değiştirmek = sadece base_url + model + api_key ortam değişkeni.
Kod değişmez. (OpenRouter bugün, vLLM yarın — ikisi de aynı API.)

API key ASLA koda/log'a girmez; yalnızca ortam değişkeninden okunur.
"""
from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass
from typing import Optional

from openai import BadRequestError, OpenAI, RateLimitError

# SDK varsayılanı 600 sn (10 dk!) + 2 otomatik retry = takılan istek yarım saat
# bekletebilir (gerçek koşuda görüldü: 'Literatür aranıyor...' ekranda kaldı).
DEFAULT_TIMEOUT = 120.0   # saniye; tekil istek için makul üst sınır
DEFAULT_RETRIES = 1

# BEDAVA MODEL RATE-LIMIT (429) — ölçülen gerçek sorun: small-cap ve kripto
# kampanyalarında 24 deneyin 11'i "429 temporarily rate-limited upstream" ile
# ATLANDI (örneklem yarıya düştü, DSR/FDR gücü zayıfladı). SDK'nın kendi retry'ı
# upstream 429'da yetmiyor: sağlayıcı saniyeler-dakikalar mertebesinde soğuma
# istiyor. Bu yüzden burada AYRI, uzun beklemeli bir retry var.
RATE_LIMIT_RETRIES = 4
RATE_LIMIT_BACKOFF = 20.0   # saniye; 20, 40, 60, 80 -> toplam ~3.3 dk sabır

# SERT (wall-clock) TIMEOUT — bkz. OpenAICompatibleClient._create.
# httpx read-timeout'u, sunucu bağlantıyı canlı tutup hiç veri göndermediğinde
# GÜVENİLİR tetiklenmiyor (gerçek koşuda görüldü: kampanya OpenRouter'a açık bir
# bağlantıda 10+ dk 0 CPU ile asılı kaldı). Çözüm: create()'i DAEMON thread'de
# çalıştır, ana thread süreyi beklesin; dolarsa TimeoutError fırlat. Thread daemon
# olduğundan asılı kalsa bile süreç çıkışını BLOKLAMAZ (ThreadPoolExecutor bunu
# yapamıyordu: non-daemon thread'leri atexit'te join edip kapanışı kilitliyordu).


@dataclass
class LLMResponse:
    text: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    model: str = ""


class OpenAICompatibleClient:
    def __init__(self, base_url: str, api_key_env: str,
                 default_headers: Optional[dict] = None,
                 timeout: float = DEFAULT_TIMEOUT) -> None:
        api_key = os.environ.get(api_key_env)
        if not api_key:
            raise RuntimeError(
                f"'{api_key_env}' ortam değişkeni yok. Key'i .env dosyasına ekle "
                f"(örn. {api_key_env}=sk-...). Key koda girmemeli.")
        self.client = OpenAI(base_url=base_url, api_key=api_key,
                             default_headers=default_headers or {},
                             timeout=timeout, max_retries=DEFAULT_RETRIES)

    def _create(self, kwargs: dict, hard_timeout: float):
        """create()'i DAEMON thread'de çalıştırıp SERT süre dolunca fırlatır.

        httpx read-timeout güvenilmez olduğundan (bkz. modül notu) tek gerçek üst
        sınır budur. Süre dolarsa çağrı arka planda bırakılır (daemon thread süreç
        çıkışını bloklamaz); çağıran hatayı yakalayıp devam eder.
        """
        box: dict = {}
        done = threading.Event()

        def _run() -> None:
            try:
                box["resp"] = self.client.chat.completions.create(**kwargs)
            except BaseException as exc:  # noqa: BLE001 — hatayı ana thread'e taşı
                box["exc"] = exc
            finally:
                done.set()

        threading.Thread(target=_run, daemon=True, name="llm-create").start()
        if not done.wait(timeout=hard_timeout + 5.0):  # httpx'e önce kendi şansı
            raise TimeoutError(
                f"LLM çağrısı {hard_timeout + 5.0:.0f}s içinde dönmedi "
                f"(sert timeout; asılı bağlantı bırakıldı).")
        if "exc" in box:
            raise box["exc"]
        return box["resp"]

    def chat(self, model: str, system: str, user: str, temperature: float = 0.7,
             force_json: bool = True, max_tokens: int = 4000,
             web_search: bool = False,
             timeout: Optional[float] = None) -> LLMResponse:
        kwargs: dict = dict(
            model=model,
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user}],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        hard_timeout = timeout if timeout is not None else DEFAULT_TIMEOUT
        if timeout is not None:
            kwargs["timeout"] = timeout   # istek bazında üst sınır (örn. literatür)
        if force_json:
            kwargs["response_format"] = {"type": "json_object"}
        if web_search:
            # OpenRouter web_search aracı (hoca örneği). SDK doğrulamasını atlamak
            # için extra_body ile gönderilir; OpenRouter aramayı kendisi yürütür.
            kwargs["extra_body"] = {"tools": [
                {"type": "openrouter:web_search",
                 "parameters": {"engine": "auto", "max_results": 5}}]}
        def _call():
            try:
                return self._create(kwargs, hard_timeout)
            except BadRequestError:
                # Bazı modeller response_format desteklemez; JSON zorlamadan tekrar
                # dene. (Timeout/ağ hatasında TEKRAR DENEME — yukarı fırlat ki çağıran
                # taraf 'literatürsüz devam' gibi kararını verebilsin.)
                if force_json:
                    kwargs.pop("response_format", None)
                    return self._create(kwargs, hard_timeout)
                raise

        # RATE-LIMIT SABRI: bedava modellerde upstream 429 sık; sabırsız davranmak
        # deney slotunu ÇÖPE ATIYOR (ölçüldü: 24 deneyin 11'i). Bekleyip yeniden
        # denemek slotu kurtarır. Diğer hatalar anında yukarı fırlar (yutulmaz).
        for attempt in range(RATE_LIMIT_RETRIES + 1):
            try:
                resp = _call()
                break
            except RateLimitError:
                if attempt >= RATE_LIMIT_RETRIES:
                    raise
                wait = RATE_LIMIT_BACKOFF * (attempt + 1)
                print(f"    [llm] rate-limit (429); {wait:.0f} sn bekleyip tekrar "
                      f"deneniyor ({attempt + 1}/{RATE_LIMIT_RETRIES})...", flush=True)
                time.sleep(wait)
        usage = resp.usage
        return LLMResponse(
            text=resp.choices[0].message.content or "",
            prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
            completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
            model=resp.model or model,
        )
