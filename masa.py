"""
Beyin Fırtınası Masası
Her sabah üç yapay zeka bir iş fikrini tartışır, masa başkanı karar yazar.
Sonuç data/ klasörüne JSON olarak kaydedilir, index.html bunu gösterir.

Koltuklar:
  1. Gemini    -> Hevesli girişimci  (GEMINI_API_KEY, ücretsiz katman)
  2. GPT       -> Şüpheci yatırımcı  (OpenAI gpt-oss, Groq ücretsiz katmanı, GROQ_API_KEY)
  3. Qwen/Llama -> Sahadaki esnaf    (Groq'taki diğer açık model, Groq ücretsiz katmanı, GROQ_API_KEY)
  4. Claude    -> Masa başkanı       (ANTHROPIC_API_KEY, opsiyonel; yoksa Gemini)

Sadece Python standart kütüphanesi kullanır.
"""

import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(ROOT, "data")
TR = timezone(timedelta(hours=3))

GEMINI_KEY = os.environ.get("GEMINI_API_KEY", "").strip()
GROQ_KEY = os.environ.get("GROQ_API_KEY", "").strip()
CLAUDE_KEY = os.environ.get("ANTHROPIC_API_KEY", "").strip()


def log(*a):
    print(*a, file=sys.stderr, flush=True)


# --------------------------------------------------------------------------
# HTTP yardımcıları
# --------------------------------------------------------------------------

def http_json(url, payload=None, headers=None, timeout=60):
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(url, data=data, method="POST" if data else "GET")
    req.add_header("Content-Type", "application/json")
    req.add_header("Accept", "application/json")
    req.add_header("User-Agent", "beyin-firtinasi-masasi")
    for k, v in (headers or {}).items():
        req.add_header(k, v)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        raw = r.read()
        enc = r.headers.get("Content-Encoding", "")
        if enc == "gzip":
            import gzip
            raw = gzip.decompress(raw)
        try:
            return json.loads(raw.decode("utf-8", errors="replace"))
        except json.JSONDecodeError:
            raise RuntimeError(f"JSON değil (HTTP {r.status}, {r.headers.get('Content-Type')}): {raw[:200]!r}")


def with_retry(fn, tries=2, wait=8):
    last = None
    for i in range(tries):
        try:
            return fn()
        except urllib.error.HTTPError as e:
            body = e.read().decode(errors="ignore")[:300]
            last = RuntimeError(f"HTTP {e.code}: {body}")
            if e.code in (429, 500, 502, 503, 504) and i < tries - 1:
                time.sleep(wait)
                continue
            raise last
        except Exception as e:  # ağ hatası
            last = e
            if i < tries - 1:
                time.sleep(wait)
                continue
    raise last


def clean(text):
    text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.S)
    return text.strip()


# --------------------------------------------------------------------------
# Sağlayıcılar
# --------------------------------------------------------------------------

_gem_models = None


def gemini_models():
    """Ücretsiz katmanda çalışan flash modellerini sırayla döndürür."""
    global _gem_models
    if _gem_models is not None:
        return _gem_models
    prefs = ["gemini-flash-latest", "gemini-flash-lite-latest"]
    found = []
    try:
        res = http_json(
            "https://generativelanguage.googleapis.com/v1beta/models?pageSize=200&key="
            + urllib.parse.quote(GEMINI_KEY), timeout=20)
        for m in res.get("models", []):
            name = m.get("name", "").replace("models/", "")
            meth = m.get("supportedGenerationMethods", [])
            bad = any(w in name for w in ("image", "tts", "live", "audio", "embedding", "exp", "robotics", "computer"))
            if "generateContent" in meth and "flash" in name and not bad:
                found.append(name)
    except Exception as e:
        log("Gemini model listesi alınamadı:", e)
    stable = sorted([n for n in found if "preview" not in n], reverse=True)
    preview = sorted([n for n in found if "preview" in n], reverse=True)
    out = []
    for n in prefs + stable + preview:
        if n not in out:
            out.append(n)
    _gem_models = out[:4]
    return _gem_models


def ask_gemini(system, messages, arama=False):
    if not GEMINI_KEY:
        raise RuntimeError("GEMINI_API_KEY yok")
    if arama:
        try:
            return _ask_gemini(system, messages, True)
        except Exception as e:
            log("Gemini Google aramalı cevap veremedi, aramasız deneniyor:", e)
    return _ask_gemini(system, messages, False)


def _ask_gemini(system, messages, arama):
    contents = [{"role": "model" if m["role"] == "assistant" else "user",
                 "parts": [{"text": m["content"]}]} for m in messages]
    payload = {"systemInstruction": {"parts": [{"text": system}]},
               "contents": contents,
               "generationConfig": {"temperature": 0.9}}
    if arama:
        payload["tools"] = [{"google_search": {}}]
    last = None
    models = gemini_models()[:2] if arama else gemini_models()
    for model in models:
        log(f"  -> Gemini {model} deneniyor{' (Google aramalı)' if arama else ''}")
        try:
            url = (f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key="
                   + urllib.parse.quote(GEMINI_KEY))
            res = with_retry(lambda: http_json(url, payload))
            parts = res["candidates"][0]["content"]["parts"]
            text = "".join(p.get("text", "") for p in parts if not p.get("thought"))
            if text.strip():
                return clean(text), model
        except Exception as e:
            last = e
            log(f"Gemini {model} olmadı:", e)
    raise RuntimeError(f"Gemini cevap vermedi: {last}")


_groq_models = None


def groq_models():
    global _groq_models
    if _groq_models is None:
        try:
            res = http_json("https://api.groq.com/openai/v1/models",
                            headers={"Authorization": f"Bearer {GROQ_KEY}"}, timeout=20)
            _groq_models = [m.get("id", "") for m in res.get("data", [])]
        except Exception as e:
            log("Groq model listesi alınamadı:", e)
            _groq_models = []
    return _groq_models


def ask_groq(system, messages, prefs, prefix):
    if not GROQ_KEY:
        raise RuntimeError("GROQ_API_KEY yok")
    candidates = (list(prefs) + [m for m in groq_models() if m.startswith(prefix)])[:4]
    seen, last = set(), None
    for model in candidates:
        if model in seen:
            continue
        seen.add(model)
        log(f"  -> Groq {model} deneniyor")
        payload = {"model": model, "temperature": 0.8,
                   "messages": [{"role": "system", "content": system}] + messages}
        if "gpt-oss" in model:
            payload["reasoning_effort"] = "low"
        try:
            res = with_retry(lambda: http_json(
                "https://api.groq.com/openai/v1/chat/completions", payload,
                headers={"Authorization": f"Bearer {GROQ_KEY}"}))
            text = res["choices"][0]["message"]["content"]
            if text and text.strip():
                return clean(text), model
        except Exception as e:
            last = e
            log(f"Groq {model} olmadı:", e)
    raise RuntimeError(f"Groq cevap vermedi: {last}")


def ask_gpt(system, messages):
    return ask_groq(system, messages, ["openai/gpt-oss-120b", "openai/gpt-oss-20b"], "openai/gpt-oss")


SKIP = ("gpt-oss", "whisper", "guard", "tts", "orpheus", "compound", "allam", "playai", "distil")


def ask_esnaf(system, messages):
    """Groq'ta gpt-oss dışında hangi sohbet modeli varsa onu oturtur (Qwen, Llama, Kimi...)."""
    others = [m for m in groq_models() if not any(w in m.lower() for w in SKIP)]
    prefs = ["qwen/qwen3.8-27b", "llama-3.3-70b-versatile", "moonshotai/kimi-k2-instruct-0905"]
    ordered = [m for m in prefs if m in others] + [m for m in others if m not in prefs]
    if not ordered:
        ordered = prefs
    return ask_groq(system, messages, ordered, "\0")


def model_adi(model):
    m = model.lower()
    for anahtar, ad in (("qwen", "Qwen"), ("llama", "Llama"), ("kimi", "Kimi"), ("deepseek", "DeepSeek"),
                        ("mistral", "Mistral"), ("gemma", "Gemma")):
        if anahtar in m:
            return ad
    return model.split("/")[-1]


def ask_claude(system, messages):
    if not CLAUDE_KEY:
        raise RuntimeError("ANTHROPIC_API_KEY yok")
    models = [os.environ.get("CLAUDE_MODEL", "").strip(), "claude-haiku-5-5", "claude-haiku-4-5"]
    last = None
    for model in [m for m in models if m]:
        payload = {"model": model, "max_tokens": 2000, "system": system, "messages": messages}
        try:
            res = with_retry(lambda: http_json(
                "https://api.anthropic.com/v1/messages", payload,
                headers={"x-api-key": CLAUDE_KEY, "anthropic-version": "2023-06-01"}))
            text = "".join(b.get("text", "") for b in res.get("content", []) if b.get("type") == "text")
            if text.strip():
                return clean(text), model
        except Exception as e:
            last = e
            log(f"Claude {model} olmadı:", e)
    raise RuntimeError(f"Claude cevap vermedi: {last}")


# --------------------------------------------------------------------------
# Masa
# --------------------------------------------------------------------------

SEATS = {
    "gemini": {
        "ad": "Gemini", "rol": "Hevesli girişimci",
        "karakter": "Fırsatları gören, enerjik ama boş hayal satmayan bir girişimcisin. "
                    "Fikri sen ortaya atarsın; itirazlar gelince fikri savunur ya da akıllıca değiştirirsin.",
        "ask": ask_gemini,
    },
    "gpt": {
        "ad": "GPT", "rol": "Şüpheci yatırımcı",
        "karakter": "Parasını kolay kolay vermeyen, soğukkanlı bir yatırımcısın. "
                    "Kim para öder, ne kadar öder, rakip kim, neden şimdi diye sorarsın. Kibar ama sert sorgularsın.",
        "ask": ask_gpt,
    },
    "esnaf": {
        "ad": "Qwen", "rol": "Sahadaki esnaf",
        "karakter": "Yıllarca İstanbul çarşılarında, pazarda, sanayide iş yapmış ayağı yere basan bir esnafsın. "
                    "İlk müşteri nereden bulunur, Türkiye'de bu iş gerçekte nasıl döner, hangi bürokrasi çıkar, "
                    "bunları sokak diliyle ama saygılı anlatırsın.",
        "ask": ask_esnaf,
    },
}

ORDER = ["gemini", "gpt", "esnaf", "gemini", "gpt", "esnaf"]

KURALLAR = """MASANIN KURALLARI (herkes uyar):
- Fikir GERÇEKÇİ olmalı. Hayal satmak, abartılı kazanç vaadi, "kolay zengin olma" yasak.
- Türkiye'de, bugünkü şartlarda uygulanabilir olmalı.
- Tam zamanlı bir işte (ör. fabrikada vardiyalı) çalışan, kod bilmeyen, lise mezunu, işini çoğunlukla telefondan yöneten bir kişi YAN İŞ olarak başlatabilmeli. Ama fikir genel olarak herkesin yapabileceği türden olsun.
- Başlangıç sermayesi düşük olmalı (tercihen 0 - 25.000 TL arası).
- İlk somut adım bu hafta atılabilmeli.
- Yasal, etik ve gerçek bir ihtiyaca dayalı olmalı.
- Türkçe konuş. Sohbet eder gibi, kısa konuş: en fazla 3 cümle (50 kelimeyi kesinlikle geçme). Başlık, madde işareti, emoji kullanma.
- Kendi adını ya da rolünü cümlenin başına yazma, doğrudan konuş."""


SORU = os.environ.get("SORU", "").strip()

KURALLAR_SORU = """MASANIN KURALLARI (herkes uyar):
- Bugün masaya bir soru getirildi; iş fikri değil, BU SORU tartışılıyor.
- GERÇEKÇİ ol. Hayal satmak, abartılı başarı vaadi yasak.
- Soran kişinin durumu: tek başına, kod bilmiyor, işini telefondan yönetiyor; kodu onun yerine bir yapay zeka asistanı yazıyor. Bütçe sıfıra yakın. Öneriler bu şartlarda gerçekten yapılabilir olmalı.
- Somut ol: isim, örnek, benzer başarılı ürünler ve neden işe yaradıkları.
- Türkçe konuş. Sohbet eder gibi, kısa konuş: en fazla 4 cümle (70 kelimeyi geçme). Başlık, madde işareti, emoji kullanma.
- Kendi adını ya da rolünü cümlenin başına yazma, doğrudan konuş."""


def gundem():
    """Güncel Türkiye ekonomi/iş haber başlıkları (tartışmaya gerçeklik zemini)."""
    url = ("https://news.google.com/rss/search?q=" + urllib.parse.quote("girişimcilik OR KOBİ OR e-ticaret OR esnaf OR \"yan gelir\" OR \"yeni iş\" when:7d")
           + "&hl=tr&gl=TR&ceid=TR:tr")
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=30) as r:
            root = ET.fromstring(r.read())
        titles = [i.findtext("title") for i in root.iter("item")][:12]
        return [t for t in titles if t]
    except Exception as e:
        log("Gündem alınamadı:", e)
        return []


def gecmis_basliklar(n=40):
    try:
        with open(os.path.join(DATA, "index.json"), encoding="utf-8") as f:
            idx = json.load(f)
        return [d.get("baslik", "") for d in idx.get("gunler", []) if d.get("tur") != "soru"][:n]
    except Exception:
        return []


ALANLAR = [
    "yemek ve gıda", "hizmet ve tamir işleri", "eğitim ve özel ders", "evcil hayvan", "yaşlı ve çocuk bakımı",
    "dijital hizmet ve sosyal medya", "el emeği ve üretim", "tarım ve bahçe", "temizlik ve ev hizmetleri",
    "otomotiv ve ulaşım", "giyim ve aksesuar", "turizm ve gezi", "spor ve sağlıklı yaşam", "geri dönüşüm ve ikinci el",
    "etkinlik ve organizasyon", "mahalle esnafına destek hizmetleri",
]


def gunun_alani(today):
    d = datetime.strptime(today, "%Y-%m-%d")
    return ALANLAR[d.toordinal() % len(ALANLAR)]


def transcript_text(turns):
    return "\n\n".join(f"{t['ad']} ({t['rol']}): {t['metin']}" for t in turns)


DUSEN = set()  # bu sabah hiç cevap veremeyen koltuklar


def speak(seat_key, system, messages, arama=False):
    """Koltuk sahibine sorar; o koltuk düşerse yedeğe geçer."""
    order = [seat_key] + [k for k in ("gpt", "esnaf", "gemini") if k != seat_key]
    order = [k for k in order if k not in DUSEN] or order
    last = None
    for k in order:
        try:
            if k == "gemini" and arama:
                text, model = ask_gemini(system, messages, arama=True)
            else:
                text, model = SEATS[k]["ask"](system, messages)
            return text, model, (k != seat_key)
        except Exception as e:
            last = e
            DUSEN.add(k)
            log(f"{k} koltuğu cevap veremedi:", e)
    raise RuntimeError(f"Masada kimse cevap veremedi: {last}")


def toplanti_soru(today):
    baglam = f"Bugün: {today}.\nMasaya getirilen soru: \"{SORU}\"\n"
    order = ORDER + ["gemini", "gpt", "esnaf"]
    turns = []
    for i, key in enumerate(order):
        seat = SEATS[key]
        system = f"Sen 'Beyin Fırtınası Masası' adlı bir tartışma masasında oturuyorsun. Rolün: {seat['rol']}. {seat['karakter']}\n\n{KURALLAR_SORU}"
        if i == 0:
            prompt = (baglam + "\nTartışmayı sen açıyorsun. Gerekirse internette araştır (benzer ürünler, ne tutuyor, ne tutmuyor). "
                      "Soruya 2-3 somut seçenekle cevap ver ve hangisini savunduğunu söyle.")
        else:
            prompt = (baglam + "\nŞimdiye kadarki konuşma:\n\n" + transcript_text(turns) +
                      "\n\nSıra sende. Bir önceki konuşmacıya doğrudan cevap ver; seçenekleri eleştir, iyileştir ya da daha iyisini öner.")
            if i >= len(order) - 3:
                prompt += " Tartışma bitmek üzere, hangi seçeneği desteklediğini net söyle."
        text, model, yedek = speak(key, system, [{"role": "user", "content": prompt}], arama=(i == 0))
        if key == "esnaf" and not yedek:
            seat["ad"] = model_adi(model)
        turns.append({"kim": key, "ad": seat["ad"], "rol": seat["rol"], "metin": text,
                      "model": model, "yedek": yedek})
        log(f"[{seat['ad']}] {text[:80]}...")
        time.sleep(4)
    return turns, []


def toplanti(today):
    if SORU:
        return toplanti_soru(today)
    haber = gundem()
    gecmis = gecmis_basliklar()
    baglam = f"Bugün: {today}.\n"
    if haber:
        baglam += "Son günlerin Türkiye gündeminden başlıklar (fikre ilham olabilir, zorunlu değil):\n- " + "\n- ".join(haber) + "\n"
    if gecmis:
        baglam += ("Daha önce masada konuşulan fikirler. Bunlara BENZEYEN hiçbir fikir getirme "
                   "(aynı ürün, aynı müşteri ya da aynı iş modeli olmasın):\n- " + "\n- ".join(gecmis) + "\n")
    baglam += f"Bugünün alanı: {gunun_alani(today)}. Fikir bu alandan olmalı.\n"

    turns = []
    for i, key in enumerate(ORDER):
        seat = SEATS[key]
        system = f"Sen 'Beyin Fırtınası Masası' adlı bir tartışma masasında oturuyorsun. Rolün: {seat['rol']}. {seat['karakter']}\n\n{KURALLAR}"
        if i == 0:
            prompt = (baglam + "\nToplantıyı sen açıyorsun. Gerekirse internette arayıp Türkiye'deki güncel bir ihtiyaca, "
                      "trende ya da boşluğa dayan. Bugün masaya TEK bir iş fikri getir: "
                      "ne olduğunu, kime satılacağını ve nasıl para kazanacağını kısaca söyle.")
        else:
            prompt = (baglam + "\nŞimdiye kadarki konuşma:\n\n" + transcript_text(turns) +
                      "\n\nSıra sende. Bir önceki konuşmacıya doğrudan cevap ver. Yeni bir fikir getirme, masadaki fikri tartış.")
            if key == "gemini":
                prompt += " İtirazları dikkate al, fikri gerekirse daha gerçekçi hale getir."
            if i >= len(ORDER) - 2:
                prompt += " Tartışma bitmek üzere, son görüşünü net söyle."
        text, model, yedek = speak(key, system, [{"role": "user", "content": prompt}], arama=(i == 0))
        if key == "esnaf" and not yedek:
            seat["ad"] = model_adi(model)
        turns.append({"kim": key, "ad": seat["ad"], "rol": seat["rol"], "metin": text,
                      "model": model, "yedek": yedek})
        log(f"[{seat['ad']}] {text[:80]}...")
        time.sleep(4)  # ücretsiz kotaları zorlamamak için
    return turns, haber


KARAR_SEMA = """{
  "baslik": "Fikrin kısa adı (en fazla 6 kelime)",
  "tek_cumle": "Fikri tek cümleyle anlatan özet",
  "nasil_para_kazanir": "1-2 cümle",
  "kime_satilir": "Hedef müşteri, 1 cümle",
  "baslangic_sermayesi": "Tahmini aralık, ör. 3.000 - 8.000 TL",
  "ilk_adim": "Bu hafta atılabilecek somut ilk adım, 1-2 cümle",
  "en_buyuk_risk": "1 cümle",
  "masanin_ortak_noktasi": "Üçünün de hemfikir olduğu nokta, 1 cümle",
  "gerceklik_puani": 7,
  "puan_gerekcesi": "Puanın kısa gerekçesi, 1 cümle",
  "karar": "SADECE şu üçünden biri, parantez veya açıklama ekleme: 'Denemeye değer', 'Şartlı evet', 'Uzak dur'"
}"""


KARAR_SEMA_SORU = """{
  "baslik": "Masanın cevabının kısa başlığı (en fazla 7 kelime)",
  "tek_cumle": "Masanın soruya tek cümlelik cevabı",
  "secenekler": [
    {"ad": "Seçeneğin adı", "aciklama": "Ne olduğu, 1-2 cümle", "arti": "En güçlü yanı, 1 cümle", "eksi": "En zayıf yanı, 1 cümle"}
  ],
  "tavsiye": "Masanın seçtiği seçenek ve neden, 2 cümle",
  "ilk_adim": "Bu hafta atılabilecek somut ilk adım, 1-2 cümle",
  "en_buyuk_risk": "1 cümle",
  "masanin_ortak_noktasi": "Herkesin hemfikir olduğu nokta, 1 cümle",
  "gerceklik_puani": 7,
  "puan_gerekcesi": "Tavsiye edilen seçeneğin yapılabilirlik puanının gerekçesi, 1 cümle",
  "karar": "SADECE şu üçünden biri: 'Denemeye değer', 'Şartlı evet', 'Uzak dur'"
}
(secenekler listesinde tartışmada konuşulan en iyi 3 seçenek olsun, tavsiye edilen ilk sırada.)"""


def karar(turns, today):
    system = ("Sen 'Beyin Fırtınası Masası'nın tarafsız başkanısın. Tartışmayı dinledin; şimdi karar raporunu yazıyorsun. "
              "Abartma, tartışmada çıkan itirazları dürüstçe yansıt. Gerçeklik puanı 1-10 arasında bir tam sayıdır "
              "(10 = yarın başlanabilir ve iş görür). Türkçe yaz.")
    sema = KARAR_SEMA_SORU if SORU else KARAR_SEMA
    soru = f"Masaya getirilen soru: {SORU}\n\n" if SORU else ""
    prompt = (f"Tarih: {today}\n\n{soru}Tartışma:\n\n{transcript_text(turns)}\n\n"
              f"Sadece aşağıdaki şemaya uyan geçerli bir JSON döndür, başka hiçbir şey yazma:\n{sema}")
    msgs = [{"role": "user", "content": prompt}]
    # Fikri Gemini açtığı için tarafsızlık adına başkanlık önce Claude'da, yoksa GPT'de
    sources = [("claude", "Claude", ask_claude), ("gpt", "GPT", ask_gpt), ("gemini", "Gemini", ask_gemini)]
    last = None
    for key, ad, fn in sources:
        try:
            text, model = fn(system, msgs)
            m = re.search(r"\{.*\}", text, re.S)
            obj = json.loads(m.group(0))
            try:
                obj["gerceklik_puani"] = max(1, min(10, int(obj.get("gerceklik_puani", 5))))
            except Exception:
                obj["gerceklik_puani"] = 5
            obj["baskan"] = ad
            obj["baskan_model"] = model
            return obj
        except Exception as e:
            last = e
            log(f"Başkan {ad} karar yazamadı:", e)
    raise RuntimeError(f"Karar yazılamadı: {last}")


def kaydet(today, turns, rapor, haber):
    os.makedirs(DATA, exist_ok=True)
    simdi = datetime.now(TR)
    kid = f"soru-{today}-{simdi.strftime('%H%M')}" if SORU else today
    kayit = {"id": kid, "tur": "soru" if SORU else "fikir", "soru": SORU, "tarih": today,
             "olusturma": simdi.isoformat(timespec="minutes"),
             "karar": rapor, "tartisma": turns, "gundem": haber}
    with open(os.path.join(DATA, f"{kid}.json"), "w", encoding="utf-8") as f:
        json.dump(kayit, f, ensure_ascii=False, indent=2)

    path = os.path.join(DATA, "index.json")
    try:
        with open(path, encoding="utf-8") as f:
            idx = json.load(f)
    except Exception:
        idx = {"gunler": []}
    idx["gunler"] = [d for d in idx["gunler"] if (d.get("id") or d.get("tarih")) != kid]
    idx["gunler"].insert(0, {"id": kid, "tur": kayit["tur"], "tarih": today, "olusturma": kayit["olusturma"],
                             "baslik": rapor.get("baslik", ""),
                             "tek_cumle": rapor.get("tek_cumle", ""),
                             "gerceklik_puani": rapor.get("gerceklik_puani"),
                             "karar": rapor.get("karar", "")})
    idx["gunler"].sort(key=lambda d: d.get("olusturma") or d["tarih"], reverse=True)
    idx["guncelleme"] = kayit["olusturma"]
    with open(path, "w", encoding="utf-8") as f:
        json.dump(idx, f, ensure_ascii=False, indent=2)
    return kid


def main():
    today = datetime.now(TR).strftime("%Y-%m-%d")
    log("Masa toplanıyor:", today, f"(SORU: {SORU})" if SORU else "(günün fikri)")
    log("Gemini anahtarı:", "var" if GEMINI_KEY else "YOK (yerine yedek oturacak)")
    log("Groq anahtarı:", "var" if GROQ_KEY else "YOK (GPT ve Llama yerine yedek oturacak)")
    log("Claude anahtarı:", "var" if CLAUDE_KEY else "yok (başkanlığı GPT yapacak)")
    turns, haber = toplanti(today)
    rapor = karar(turns, today)
    kid = kaydet(today, turns, rapor, haber)
    log("Bitti:", rapor.get("baslik"))
    if os.environ.get("GITHUB_OUTPUT"):
        with open(os.environ["GITHUB_OUTPUT"], "a") as f:
            f.write(f"kayit={kid}\n")


if __name__ == "__main__":
    main()
