# Binance USDT Signal Bot – Faz 1 (Mevcut Durum)

Bu doküman, şu an elinde bulunan kod ve dokümanlara göre **Faz 1** kapsamını tanımlar.  
Faz 1, botun **temel iskeletini ve kural tabanlı karar mekanizmasını** ayağa kaldırmayı hedefler.

---

## 1. Amaç

Faz 1’in amacı:

- Binance USDT paritelerini 24h istatistiklerine göre **prefilter** etmek,
- Uygun bulunan coinler için:
  - Multi-timeframe (özellikle 15m, 1h) OHLCV verisini çekmek,
  - Tüm indikatörleri hesaplamak,
  - Trend / Oscillator / Volume / Price Action bloklarından **skor ve gerekçe** üretmek,
  - `STRONG_BUY` ve `ULTRA_BUY` etiketleriyle sinyal üretmek,
- Sinyalleri:
  - Konsola loglamak,
  - CSV’ye kaydetmek,
  - Telegram üzerinden sinyal göndermektir.

---

## 2. Mevcut Modüller ve Rolleri

### 2.1. `main.py`

- Ana loop ve market tarama akışı.
- Görevleri:
  - `data_fetcher.fetch_24h_tickers` ile 24h ticker verisini çekmek.
  - `data_fetcher.parse_ticker_data` ile ticker’ı normalize etmek.
  - `config` prefilter kurallarına göre sembolleri süzmek:
    - Min 24h hacim,
    - Min fiyat,
    - 24h değişim aralığı,
    - Cooldown kontrolü.
  - Prefilter sonrası sembolleri hacme göre kısıtlayıp (`MAX_SYMBOLS_PER_SCAN`), `analyzer.analyze_symbol` ile analiz ettirmek.
  - Sinyalleri:
    - CSV’ye yazmak (`logger.log_signal_to_csv`),
    - Telegram’a göndermek (`telegram_bot`),
    - Konsola yazmak.

### 2.2. `analyzer.py`

- Tek bir sembol için çok adımlı analiz:

  1. Prefilter (ikinci katman – güvenlik / tutarlılık).
  2. `data_fetcher.fetch_multi_timeframe_klines` ile 15m, 1h (ve 4h) verisini almak.
  3. OHLCV dizilerini üretmek.
  4. `indicators.py` üzerinden tüm indikatörleri **gerçek formüllerle** hesaplatmak.
  5. Son bar değerlerini ve MACD histogram trendini çıkarmak.
  6. 1h timeframe’de EMA20/EMA50 yapısını kontrol ederek multi-timeframe trend onayı yapmak.
  7. `price_action.analyze_price_action` ile mum formasyonları, EMA20 breakout, volume spike, no collapse vb. sinyalleri almak.
  8. `rules.compute_*_block` fonksiyonlarıyla:
     - Trend Block
     - Oscillator Block
     - Volume Block
     - Price Action Block
     skorlarını hesaplamak.
  9. `rules.decide_signal_label` ile final etiketi ve toplam skoru üretmek.
  10. Sonucu `SignalResult.__dict__` formatında döndürmek.

### 2.3. `rules.py`

- Tamamen **kural tabanlı** skor sistemi (ML yok):

  - `compute_trend_block`  
    ADX + DI, EMA hizalanması, MACD histogram, momentum & AO, 1h trend onayı.
  - `compute_osc_block`  
    RSI, Stoch K, CCI, Stoch RSI, Williams %R, Ultimate Oscillator.
  - `compute_volume_block`  
    OBV trendi, Bull/Bear Power, volume spike uyumu.
  - `compute_price_action_block`  
    Long lower wick, strong green candle, no collapse, EMA20 break, volume onayı.
  - `decide_signal_label`  
    `ULTRA_BUY` ve `STRONG_BUY` eşikleri; trend/osc/vol+pa minimum skorları; RSI sınırı ve 1h trend gereklilikleri.

### 2.4. `config.py`

- Tüm sabit parametreler:

  - Binance API URL’leri ve sembol filtresi (`USDT`).
  - Prefilter limitleri (min hacim, min fiyat, 24h değişim aralığı).
  - Cooldown süresi, maksimum sembol sayısı.
  - Telegram ayarları.
  - Logging ayarları.
  - İndikatör parametreleri (EMA, ADX, MACD, RSI vb.).
  - Price Action ve backtest parametreleri.

---

## 3. Tamamlanması / Kontrol Edilmesi Gereken Modüller (Faz 1 Kapsamında Zorunlu)

Bu dosyalar olmadan Faz 1 botu çalışmaz; bu fazda **tam üretim kalitesinde** yazılmaları gerekir:

1. `data_fetcher.py`
   - Fonksiyonlar:
     - `async def fetch_24h_tickers(session) -> list`
     - `def parse_ticker_data(raw_ticker) -> dict`
     - `async def fetch_multi_timeframe_klines(session, symbol: str) -> dict`

2. `indicators.py`
   - Tüm indikatör fonksiyonları:
     - EMA, ADX(+DI), MACD, Momentum, AO, RSI, Stoch K, CCI, Stoch RSI, Williams %R, Ultimate Oscillator, OBV, Bull/Bear Power, OBV trend.

3. `price_action.py`
   - `def analyze_price_action(opens, highs, lows, closes, volumes, ema20_values) -> dict`
   - Dönen sözlükte en az:
     - `long_lower_wick`, `strong_green`, `no_collapse`, `ema20_break`, `volume_spike`, `min_volume`.

4. `telegram_bot.py`
   - `def format_signal_message(signal: dict) -> str`
   - `async def send_telegram_message(text: str) -> None`

5. `logger.py`
   - `def setup_logger() -> logging.Logger`
   - `def log_signal_to_csv(path: str, signal: dict, extra_fields: dict | None = None) -> None`

---

## 4. Faz 1’in Teslim Kriterleri

- Kod başarıyla:
  - Binance’ten 24h verisi çekiyor,
  - Prefilter uyguluyor,
  - 15m + 1h verisiyle indikatörler hesaplıyor,
  - Trend / Oscillator / Volume / Price Action blok skoru çıkarıyor,
  - `STRONG_BUY` ve `ULTRA_BUY` sinyalleri üretebiliyor.
- Sinyaller:
  - Konsolda düzgün loglanıyor,
  - `signals_log.csv` dosyasına yazılıyor.
- (İsteğe bağlı) Telegram:
  - `ENABLE_TELEGRAM = True` olduğunda mesaj gönderiyor.
- Bot:
  - `main.py` ile çalıştırılabilir,
  - Hata durumunda kapanmadan toparlanmaya çalışıyor,
  - Systemd servisine uygun şekilde uzun süreli koşuya hazır.

Faz 1 tamamlandığında, sistem **temel stratejiyi** uygulayabilen stabil bir iskelete sahip olur. Faz 2 ve Faz 3, bu iskelet üzerine stratejik iyileştirmeler ve genişlemeler getirecektir.