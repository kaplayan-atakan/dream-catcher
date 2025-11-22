# Binance USDT Signal Bot – Faz 2 (Strateji Derinleştirme)

Bu doküman, **Faz 1 tamamlanmış** kabul edilerek, Faz 2’de yapılacak strateji ve yapı iyileştirmelerini tanımlar.  
Faz 2’nin temel kaynağı, müşterinin gönderdiği **[client-message.md](client-message.md)** dosyasındaki blok tanımlarıdır.

Faz 2’nin ana amacı:  
**Mevcut kural tabanlı sistemi, client-message.md’de tarif edilen stratejiye daha sıkı hizalamak ve her bloğu daha şeffaf/ayrıntılı hale getirmek.**

---

## 1. Trend Analysis Block – Derinleştirme

Kaynak: `client-message.md > 1. Trend Analysis Block`

### 1.1. Hedef

Faz 1’de trend bloğu temel olarak çalışıyor; Faz 2’de:

- Trend bloğunun **içsel durumunu** daha detaylı raporlamak,
- ADX + DI, MACD hist, Momentum, AO ve EMA20/EMA50 yapısını **ayrı alt sinyaller** olarak dışarı açmak,
- 1h timeframe konfirmasyonunu son kullanıcıya daha okunur hale getirmek.

### 1.2. Teknik Değişiklikler

1. `rules.compute_trend_block` içinde:
   - İçeride zaten mevcut olan kuralları, `client-message.md` ile birebir label’lara bağla:
     - Örnek:
       - `"Trend: ADX strong, DI+>DI-"` → Client dokümanındaki “ADX above 20 + DI+ > DI–” maddesine referans olarak yorumlanabilir.
   - Faz 2’de eklenmesi önerilen alanlar:
     - `BlockScore`’a opsiyonel bir `details` dict eklenebilir (backward compatible kalmak istiyorsan yeni bir yapı da tanımlanabilir):
       ```python
       {
         "adx": last_adx,
         "plus_di": last_plus_di,
         "minus_di": last_minus_di,
         "macd_hist": last_macd_hist,
         "macd_hist_rising": macd_hist_rising,
         "momentum": last_momentum,
         "ao": last_ao,
         "ema20": last_ema20,
         "ema50": last_ema50,
         "price_above_ema20": bool,
         "ema_alignment_ok": bool,
         "mtf_trend_confirmed": bool
       }
       ```
     - Böylece Telegram mesajı veya logta “Trend metrics” blok halinde gösterilebilir.

2. 1h timeframe konfirmasyonu:
   - `analyzer.py` içinde zaten:
     - `mtf_trend_confirmed` ve `htf_price_above_ema` hesaplanıyor.
   - Faz 2’de:
     - Bu alanlar, `SignalResult` içine opsiyonel olarak eklenebilir (örneğin `extra` alanında veya ileride kullanılmak üzere).

---

## 2. Oscillator Block – Derinleştirme

Kaynak: `client-message.md > 2. Oscillator Block`

### 2.1. Hedef

- RSI, Stoch K, CCI, Stoch RSI, Williams %R, UO için:
  - Faz 1’de zaten skor üretiliyor.
  - Faz 2’de, **müşterinin tanımladığı aralıklar** ile uyumu daha net yansıtma.

### 2.2. Teknik Değişiklikler

1. `rules.compute_osc_block` içinde:
   - Client dokümanındaki cümleleri “neden” mesajlarına yansıt:
     - Örnek:
       - “Values between 45–65 indicate a healthy, sustainable trend.”  
         → Zaten `RSI_HEALTHY_MIN` ve `RSI_HEALTHY_MAX` ile eşleştirilmiş; reason string’ini daha açıklayıcı yap:
         - `"Osc: RSI=52.3 in 45–65 healthy range (sustainable trend)"`
   - `Stochastic K` için:
     - Client: “Crossing above 50 is considered bullish.”
     - Mevcut kural zaten `StochK > 50` için +1 veriyor; reason’ı bunu yansıtacak şekilde güncelle:
       - `"Osc: StochK=68.2 above 50 (bullish range)"`

2. Gelişmiş raporlama:
   - Trend bloğuna benzer şekilde, opsiyonel bir `details` dict:
     ```python
     {
       "rsi": last_rsi,
       "stoch_k": last_stoch_k,
       "cci": last_cci,
       "stoch_rsi": last_stoch_rsi,
       "williams_r": last_williams_r,
       "uo": last_uo,
       "rsi_healthy": bool,
       "stoch_bullish": bool,
       "cci_strong_bull": bool,
       ...
     }
     ```

---

## 3. Volume & Power Block – Derinleştirme

Kaynak: `client-message.md > 3. Volume & Power Block`

### 3.1. Hedef

- Hacim bloğunda:
  - OBV trendi,
  - Volume spike,
  - Bull/Bear Power
  zaten kullanılıyor.
- Faz 2’de:
  - Müşterinin “OBV 10-bar trend” ve “Volume spike 1.5x” ifadelerini metrik seviyesinde raporlamak.

### 3.2. Teknik Değişiklikler

1. `price_action.analyze_price_action`:
   - `volume_spike` hesaplanırken `config.VOLUME_SPIKE_MULTIPLIER` ve `config.VOLUME_LOOKBACK` zaten kullanılmalı.
   - Faz 2’de, volume spike’ın **kaç kat** olduğunu hesaplayıp `pa_signals` içine ekleyebilirsin:
     ```python
     pa_signals["volume_spike_factor"] = current_volume / avg_volume
     ```

2. `indicators.is_obv_uptrend`:
   - Halihazırda `OBV_TREND_LOOKBACK` (=10) üzerinden bir trend kontrolü yapmalı.
   - Faz 2’de, OBV değişim oranını da döndürmek düşünülebilir (ama imza bozmadan, örneğin ayrı bir helper fonksiyonla).

3. `rules.compute_volume_block`:
   - Reason string’lerini client dokümanı terminolojisine daha çok yaklaştır:
     - `"Vol: OBV in uptrend over last 10 bars (accumulation)"`
     - `"Vol: Volume spike 1.8x above 20-bar average"`

---

## 4. Price Action Block – Derinleştirme

Kaynak: `client-message.md > 4. Price Action Block`

### 4.1. Hedef

- Price action tarafında:
  - Long lower wick (hammer tipi),
  - Strong green candle,
  - No recent collapse (96 bar),
  - EMA20 breakout,
  - Minimum volume condition
  zaten `rules.compute_price_action_block`’ta flag’ler olarak kullanılıyor.
- Faz 2’de, bu flag’leri **daha net ölçümlerle** desteklemek.

### 4.2. Teknik Değişiklikler

1. `price_action.analyze_price_action` fonksiyonunda:

   - Long lower wick:
     - Aşağıdakileri hesaplayıp isteğe bağlı `details` alanına ekle:
       - Wick uzunluğu yüzdesi,
       - Body yüzdesi (`config.MIN_BODY_PCT` eşiklerine göre).
   - Strong green candle:
     - Body’nin toplam range’e oranı,
     - Kapanışın open’a göre yüzdesi.
   - No recent collapse:
     - Son 96 bar içinde max düşüş yüzdesi.
   - EMA20 breakout:
     - Önceki bar EMA20 altında, son bar EMA20 üstünde mi?
   - Minimum volume:
     - `volume >= MIN_BAR_VOLUME_USDT` koşulu.

2. `rules.compute_price_action_block` reason’ları, client dokümanı cümlelerine benzet:
   - `"PA: Hammer-type candle (long lower wick, buyers absorbed dip)"`
   - `"PA: No sharp dump in last 96 candles"`

---

## 5. Prefilter & Multi-Timeframe – İnce Ayarlar

Kaynak: `client-message.md > 5. Prefilter Layer`, `6. Multi-Timeframe Structure`

Faz 2’de:

- Prefilter kuralları zaten `config.py` üzerinden net tanımlı:
  - 24h volume ≥ 5M,
  - Fiyat ≥ 0.02,
  - 24h change –15% ile +20% arası,
  - Cooldown 60 dk,
  - Top-volume limiting (MAX_SYMBOLS_PER_SCAN).

- İyileştirme:
  - Prefilter sonuçlarını loglarda daha şeffaf göstermek:
    - Kaç coin elendi:
      - Hacim yüzünden,
      - Fiyat yüzünden,
      - Değişim aralığı yüzünden,
      - Cooldown yüzünden.

- Multi-timeframe:
  - Şimdilik 15m & 1h aktif, 4h ileride (Faz 3’te) kullanılacak; Faz 2’de sadece koda **hazırlık** (4h verisini de `data_fetcher` içine almak, analyzer’da istersen yorumlayıp loglamak) yapılabilir.

---

## 6. Faz 2 Teslim Kriterleri

- Trend, Oscillator, Volume, Price Action bloklarının **sebep metinleri** (reasons) client-message.md ile daha yüksek oranda örtüşüyor.
- Opsiyonel (ama önerilen) olarak:
  - Her blok için ayrıntılı `details` yapısı mevcut.
  - Bu yapı Telegram mesaj formatında ve CSV logta istenirse kullanılabilir.
- Prefilter ve multi-timeframe kullanımı client dokümanındaki tanımlarla uyumlu ve loglarda görülebilir.

Faz 2 tamamlandığında, botun verdiği her sinyal **“neden bu sinyal üretildi?”** sorusuna detaylı, stratejiye uygun yanıt verebilecek seviyeye gelir.