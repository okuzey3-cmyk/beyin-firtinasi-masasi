# Beyin Fırtınası Masası

Her sabah 07:07'de (Türkiye saati) üç yapay zeka bir iş fikrini tartışır, masa başkanı karar yazar.

| Koltuk | Rol | Kaynak |
|---|---|---|
| Gemini | Hevesli girişimci | Google AI Studio (ücretsiz) — `GEMINI_API_KEY` |
| GPT | Şüpheci yatırımcı | GitHub Models (ücretsiz) |
| DeepSeek | Sahadaki esnaf | GitHub Models (ücretsiz) |
| Claude | Masa başkanı | `ANTHROPIC_API_KEY` (opsiyonel; yoksa Gemini başkanlık eder) |

Sonuçlar `data/` klasörüne kaydedilir, `index.html` bunları gösterir.
Elle çalıştırmak için: Actions → Sabah Toplantısı → Run workflow.
