 # Binance USDT Signal Bot

Bu proje, Binance üzerindeki **USDT paritelerini** tarayıp, tamamen **kural tabanlı** (NO ML/AI) bir sistemle:

- **Trend**
- **Oscillator**
- **Volume & Power**
- **Price Action**
- **Prefilter (24h istatistikleri)**
- **Multi-timeframe yapı (15m / 1h / 4h)**

katmanlarını birlikte değerlendirerek, özellikle **`STRONG_BUY`** ve **`ULTRA_BUY`** sinyalleri üreten bir bottur.

Tüm karar mantığı, müşteriden gelen dokümanlar (`docs/client-message.md`, `docs/gosterge-ozet.md`) ve mevcut Python modülleri (`analyzer.py`, `main.py`, `config.py`, `rules.py`) üzerine inşa edilmiştir.

---

## 1. Proje Yapısı

Proje kök dizini (sen kendi ortamında `project/` veya benzeri bir klasör kullanabilirsin):

```text
project/
├─ src/
│  ├─ main.py
│  ├─ analyzer.py
│  ├─ data_fetcher.py        # (tamamlanması gereken / kontrol edilecek)
│  ├─ indicators.py          # (tamamlanması gereken)
│  ├─ price_action.py        # (tamamlanması gereken)
│  ├─ rules.py               # VAR (kural tabanlı skor sistemi)
│  ├─ telegram_bot.py        # (tamamlanması gereken)
│  ├─ logger.py              # (tamamlanması gereken)
│  └─ config.py
│
├─ docs/
│  ├─ gosterge-ozet.md       # Gösterge eşikleri + örnek yorum
│  ├─ client-message.md      # Müşterinin blok bazlı strateji açıklaması
│  └─ bot-init-doc.txt       # Sunucuda ayağa kaldırma rehberi
│
└─ signals_log.csv           # Otomatik oluşturulur (log_module.log_signal_to_csv ile)
```

> Not: Senin eline ilk başta eksik kodlar geçmiş; bu README, **tamamlanacak dosyaları** net şekilde işaretleyerek kodlama ajanının işi devralmasını amaçlar.

---

## 2. Modüllerin Rolü

### 2.1. `main.py`

- Botun **giriş noktası** ve sonsuz döngülü tarama mekanizması.
- Görevleri:
  - `aiohttp` ile **paylaşımlı HTTP session** oluşturmak.
  - Her döngüde:
    1. `scan_market(session)` ile piyasayı taramak.
    2. Üretilen sinyalleri:
       - Cooldown tablosuna (`last_signal_times`) eklemek,
       - CSV’ye loglamak (`logger.log_signal_to_csv`),
       - İsteğe bağlı Telegram’a göndermek (`telegram_bot`),
       - Konsola insan okunur şekilde yazmak.
  - Hata durumlarında bekleyip (örn. 30 saniye) toparlanmaya çalışmak.

Ana fonksiyonlar:

- `async def scan_market(session: aiohttp.ClientSession) -> List[dict]`
- `async def main_loop()`
- `def main()`

---

### 2.2. `analyzer.py`

- **Tek bir sembolü** derinlemesine analiz eder.
- Girdi: `symbol_data` (24h ticker’den türetilmiş prefilter sonrası veri).
- Adımlar:

1. **Prefilter (ek güvenlik):**
   - 24h volume, fiyat, 24h değişim gibi metrikleri **tekrar** kontrol eder:
     ```python
     if symbol_data['quote_volume'] < config.MIN_24H_QUOTE_VOLUME: return None
     if symbol_data['price'] < config.MIN_PRICE_USDT: return None
     if not (config.MIN_24H_CHANGE <= symbol_data['price_change_pct'] <= config.MAX_24H_CHANGE): return None
     ```

2. **Multi-timeframe kline çekme:**
   - `data_fetcher.fetch_multi_timeframe_klines(session, symbol)`
   - Beklenen: `config.TIMEFRAMES` (default: `["15m", "1h", "4h"]`) için OHLCV listeleri.
   - Ana timeframe: `config.MAIN_TIMEFRAME` (default: `"15m"`); burada min 200 bar şartı var.

3. **OHLCV dizilerini çıkartır:**
   - `opens`, `highs`, `lows`, `closes`, `volumes`.

4. **Tüm indikatörleri hesaplar (`indicators.py` üzerinden):**

   - Moving averages:
     - `ema20_values = indicators.ema(closes, config.EMA_FAST)` (20)
     - `ema50_values = indicators.ema(closes, config.EMA_SLOW)` (50)

   - Trend:
     - `adx_values, plus_di_values, minus_di_values = indicators.adx(...)`
     - `macd_line, signal_line, macd_hist = indicators.macd(closes)`
     - `momentum_values = indicators.momentum(closes, config.MOMENTUM_PERIOD)` (10)
     - `ao_values = indicators.awesome_oscillator(highs, lows)`

   - Oscillators:
     - `rsi_values = indicators.rsi(closes, config.RSI_PERIOD)`
     - `stoch_k_values = indicators.stochastic_k(highs, lows, closes, config.STOCH_K_PERIOD)`
     - `cci_values = indicators.cci(highs, lows, closes, config.CCI_PERIOD)`
     - `stoch_rsi_values = indicators.stochastic_rsi(closes)`
     - `williams_r_values = indicators.williams_r(highs, lows, closes, config.WILLIAMS_PERIOD)`
     - `uo_values = indicators.ultimate_oscillator(highs, lows, closes, config.UO_PERIODS)`

   - Volume:
     - `obv_values = indicators.obv(closes, volumes)`
     - `bull_power_values, bear_power_values = indicators.bull_bear_power(highs, lows, closes)`
     - `obv_uptrend = indicators.is_obv_uptrend(obv_values, config.OBV_TREND_LOOKBACK)`

5. **Son bar (en güncel mum) değerlerini çıkartır.**
   - Örn. `last_rsi`, `last_ema20`, `last_adx`, `last_macd_hist` vb.

6. **MACD histogramın son X bar boyunca yükselişte olup olmadığını kontrol eder.**
   - `config.MACD_HIST_RISING_BARS` kadar.

7. **1h timeframe trend doğrulama:**
   - 1h kapanışlar üzerinden EMA20 ve EMA50 hesaplatarak:
     - `htf_price_above_ema`: Fiyat son 1h kapanışta EMA20 üzerinde mi?
     - `mtf_trend_confirmed`: `close > EMA20 > EMA50` mi?

8. **Price action analizi (`price_action.analyze_price_action`)**
   - Mum formasyonları, EMA20 breakout, volume spike, no-collapse, minimum volume gibi sinyallerin boolean flag’lerini döndürmelidir.

9. **Blok skorları (`rules.py`):**
   - `trend_block = rules.compute_trend_block(...)`
   - `osc_block = rules.compute_osc_block(...)`
   - `vol_block = rules.compute_volume_block(...)`
   - `pa_block = rules.compute_price_action_block(pa_signals)`

10. **Final sinyal kararı:**
    - `signal_result = rules.decide_signal_label(...)`
    - `SignalResult` (dataclass) döner, etiket: `"NONE"`, `"STRONG_BUY"`, `"ULTRA_BUY"`.

11. **Market verilerini de sonuca ekler:**
    ```python
    signal_result.price = symbol_data['price']
    signal_result.price_change_pct = symbol_data['price_change_pct']
    signal_result.quote_volume = symbol_data['quote_volume']
    ```

12. `signal_result.label != "NONE"` ise `signal_result.__dict__` döner, aksi halde `None`.

---

### 2.3. `rules.py` (Kural Tabanlı Skorlama – VAR)

Bu dosya tamamen **kural tabanlı** bir skor sistemi içerir, ML yoktur.

#### 2.3.1. Veri Yapıları

```python
@dataclass
class BlockScore:
    score: int
    reasons: List[str]

@dataclass
class SignalResult:
    symbol: str
    trend_score: int
    osc_score: int
    vol_score: int
    pa_score: int
    total_score: int
    label: str
    reasons: List[str]
    rsi: float
    price: float = 0
    price_change_pct: float = 0
    quote_volume: float = 0
```

#### 2.3.2. Trend Block

`compute_trend_block(...) -> BlockScore`

Kriterler:

- **ADX + DI**:
  - `adx >= ADX_STRONG_TREND` ve `DI+ > DI-` → +2 puan
  - `adx >= 0.7 * ADX_STRONG_TREND` ve `DI+ > DI-` → +1 puan

- **EMA Yapısı**:
  - `price > ema20 > ema50` → +1
  - `price > ema20` → +0.5

- **MACD Histogram**:
  - `macd_hist > 0` ve `macd_hist_rising` → +1.5
  - `macd_hist > 0` → +0.5

- **Momentum & AO**:
  - Momentum > 0 ve AO > 0 → +1
  - Biri pozitif → +0.5

- **Multi-timeframe (1h) onayı**:
  - `mtf_trend == True` → +1

Top 3 sebep `reasons` listesine alınır.

#### 2.3.3. Oscillator Block

`compute_osc_block(...) -> BlockScore`

- RSI, Stoch K, CCI, Stoch RSI, Williams %R, Ultimate Oscillator, `config` parametreleri ile:
  - RSI 45–65 → +1
  - RSI 30–45 → +0.5 (oversold bounce)
  - StochK > 50 & range içinde → +1
  - StochK nötr → +0.5
  - CCI > 100 → +1
  - CCI > 0 → +0.5
  - Williams %R > WILLIAMS_BULLISH → +1
  - vs.

Top 3 sebep tutulur.

#### 2.3.4. Volume & Power Block

`compute_volume_block(...) -> BlockScore`

- OBV uptrend → +1.5
- Volume spike → +1
- Bull power > 0 & Bear power < 0 → +1.5
- Sadece bull power > 0 → +0.5
- OBV + volume spike birlikte → +0.5 bonus

#### 2.3.5. Price Action Block

`compute_price_action_block(pa_signals: dict) -> BlockScore`

`pa_signals` sözlüğünden okur:

- `long_lower_wick` → +1.5
- `strong_green` → +1
- `no_collapse` → +1
- `ema20_break` → +1.5
- `volume_spike` & `min_volume` → +1
- Sadece `min_volume` → +0.5

#### 2.3.6. Final Sinyal

`decide_signal_label(...) -> SignalResult`

- `total_score = trend + osc + vol + pa`
- `vol_pa_combined = vol + pa`

**ULTRA_BUY koşulları:**

- `total_score >= ULTRA_BUY_SCORE` (config)
- `trend_score >= ULTRA_BUY_MIN_TREND`
- `osc_score >= ULTRA_BUY_MIN_OSC`
- `vol_pa_combined >= ULTRA_BUY_MIN_VOL_PA`
- `rsi_value <= ULTRA_BUY_MAX_RSI`
- `htf_trend_ok == True` (1h EMA yapısı onaylı)

**STRONG_BUY koşulları:**

- `total_score >= STRONG_BUY_SCORE`
- `trend_score >= STRONG_BUY_MIN_TREND`
- `osc_score >= STRONG_BUY_MIN_OSC`
- `vol_pa_combined >= STRONG_BUY_MIN_VOL_PA`

Aksi durumda `label = "NONE"`.

---

### 2.4. `config.py`

Tüm sabitler ve eşik değerleri burada:

- Binance URL, sembol son eki, timeframeler:
  - `BINANCE_BASE_URL = "https://api.binance.com"`
  - `SYMBOL_FILTER_SUFFIX = "USDT"`
  - `TIMEFRAMES = ["15m", "1h", "4h"]`
  - `MAIN_TIMEFRAME = "15m"`

- Prefilterler:
  - `MIN_24H_QUOTE_VOLUME = 5_000_000`
  - `MIN_PRICE_USDT = 0.02`
  - `MIN_24H_CHANGE = -15.0`
  - `MAX_24H_CHANGE = 20.0`

- Cooldown ve limits:
  - `COOLDOWN_MINUTES = 60`
  - `MAX_SYMBOLS_PER_SCAN = 50`

- Telegram:
  - `ENABLE_TELEGRAM`, `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID`

- Logging:
  - `LOG_CSV_PATH`, `LOG_LEVEL`

- Skor & sinyal eşikleri:
  - `ULTRA_BUY_SCORE`, `STRONG_BUY_SCORE`, min-trend/min-osc/min-volpa, `ULTRA_BUY_MAX_RSI`.

- İndikatör parametreleri:
  - EMA, ADX, MACD, Momentum, RSI, Stoch, CCI, Williams, UO, OBV `lookback` vs.

- Price Action & Backtest parametreleri.

---

### 2.5. Eksik (veya Tamamlanması Gereken) Modüller

Bu modüller, mevcut kodlardan ve dokümanlardan türetilerek **kodlama ajanı** tarafından yazılmalıdır.

#### 2.5.1. `data_fetcher.py`

Beklenen fonksiyonlar:

- `async def fetch_24h_tickers(session) -> list`
  - Binance REST: `/api/v3/ticker/24hr`
  - `SYMBOL_FILTER_SUFFIX` ile `USDT` pariteleri filtrelenmeli.

- `def parse_ticker_data(raw_ticker) -> dict`
  - Dönüş:
    ```python
    {
      "symbol": str,
      "price": float,
      "quote_volume": float,
      "price_change_pct": float
    }
    ```

- `async def fetch_multi_timeframe_klines(session, symbol: str) -> dict`
  - En az `TIMEFRAMES` içindeki her timeframe için:
    - `/api/v3/klines?symbol=XXXUSDT&interval=15m` gibi endpoint’lerden OHLCV verisi.
  - Dönüş:
    ```python
    {
      "15m": [{"open":..., "high":..., "low":..., "close":..., "volume":...}, ...],
      "1h": [...],
      "4h": [...]
    }
    ```

#### 2.5.2. `indicators.py`

İmzalar `analyzer.py` ile tam uyumlu olmalı:

- `ema(values, period)`
- `adx(highs, lows, closes, period) -> (adx_list, plus_di_list, minus_di_list)`
- `macd(closes) -> (macd_line, signal_line, macd_hist)`
- `momentum(closes, period)`
- `awesome_oscillator(highs, lows)`
- `rsi(closes, period)`
- `stochastic_k(highs, lows, closes, period)`
- `cci(highs, lows, closes, period)`
- `stochastic_rsi(closes)`
- `williams_r(highs, lows, closes, period)`
- `ultimate_oscillator(highs, lows, closes, (short, mid, long))`
- `obv(closes, volumes)`
- `bull_bear_power(highs, lows, closes) -> (bull_power_list, bear_power_list)`
- `is_obv_uptrend(obv_values, lookback) -> bool`

NaN yönetimi, `analyzer.py`de yapılan `if value is not np.nan` kontrolüyle uyumlu tasarlanmalı (numpy NaN).

#### 2.5.3. `price_action.py`

Beklenen ana fonksiyon:

```python
def analyze_price_action(opens, highs, lows, closes, volumes, ema20_values) -> dict:
    ...
```

Dönüş dict’inde en azından şu key’ler olmalı (rules.py bunu bekliyor):

- `"long_lower_wick": bool`
- `"strong_green": bool`
- `"no_collapse": bool`
- `"ema20_break": bool`
- `"volume_spike": bool`
- `"min_volume": bool`

Parametreler:

- `config.MIN_BODY_PCT`
- `config.COLLAPSE_MAX_DROP_PCT`
- `config.COLLAPSE_LOOKBACK_BARS`
- `config.MIN_BAR_VOLUME_USDT`
- `config.VOLUME_SPIKE_MULTIPLIER`
- `config.VOLUME_LOOKBACK`

Bu kurallar, `docs/client-message.md` içindeki **Price Action Block** tanımına uyumlu tasarlanmalı.

#### 2.5.4. `telegram_bot.py`

Beklenen fonksiyonlar (main.py’den):

- `def format_signal_message(signal: dict) -> str`
  - Örn.:
    - Sembol, label, fiyat, 24h değişimi, 24h volume, trend/osc/vol/pa/total skorları ve en önemli 3–5 reason.

- `async def send_telegram_message(text: str) -> None`
  - `config.TELEGRAM_BOT_TOKEN` ve `config.TELEGRAM_CHAT_ID` kullanarak Telegram Bot API’ye POST atar.
  - Hata yönetimi ve basit retry eklenebilir.

#### 2.5.5. `logger.py`

Beklenen fonksiyonlar:

- `def setup_logger() -> logging.Logger`
  - Konsol log’u, isteğe bağlı file handler,
  - `config.LOG_LEVEL`’e göre seviye ayarı.

- `def log_signal_to_csv(path: str, signal: dict, extra_fields: dict | None = None) -> None`
  - `signals_log.csv` dosyasına append.
  - Kolon örnekleri:
    - `timestamp`, `symbol`, `label`, `trend_score`, `osc_score`, `vol_score`, `pa_score`, `total_score`, `rsi`, `price`, `change_24h`, `quote_vol_24h`, vb.

---

## 3. Çalıştırma ve Sunucu Kurulumu

Detaylı kurulum rehberi: `docs/bot-init-doc.txt`.

Özet:

### 3.1. Python

- Önerilen sürümler: **Python 3.10** veya **3.11**.

### 3.2. Gerekli Paketler

`requirements.txt` örneği:

```text
aiohttp
numpy
pandas
python-dateutil
requests
python-dotenv
websockets
```

Kurulum:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 3.3. Botu Çalıştırma

```bash
cd project/src
python main.py
```

---

## 4. Systemd ile 7/24 Çalıştırma (Örnek)

```ini
# /etc/systemd/system/binancebot.service

[Unit]
Description=Binance Signal Bot
After=network.target

[Service]
User=root
WorkingDirectory=/root/project/src
ExecStart=/root/project/venv/bin/python main.py
Restart=always

[Install]
WantedBy=multi-user.target
```

Aktifleştirme:

```bash
sudo systemctl daemon-reload
sudo systemctl enable binancebot
sudo systemctl start binancebot
journalctl -u binancebot -f
```

---

## 5. Kodlama Ajanı İçin Net Görev Listesi

1. `data_fetcher.py`  
   - Binance REST (24h ticker + multi-timeframe klines) entegrasyonunu yaz.
   - `analyzer.py` ve `main.py`’deki çağrılarla birebir uyumlu imzalar.

2. `indicators.py`  
   - Tüm indikatör fonksiyonlarını `analyzer.py`’de kullanılan isim ve parametrelerle implemente et.
   - Hesaplamalarda numpy kullan, NaN’leri güvenli yönet.

3. `price_action.py`  
   - `analyze_price_action(...) -> dict` fonksiyonunu yaz.
   - `rules.compute_price_action_block` tarafından beklenen flag’leri oluştur.

4. `telegram_bot.py`  
   - `format_signal_message(signal)` ve `send_telegram_message(text)` fonksiyonlarını yaz.

5. `logger.py`  
   - `setup_logger()` ve `log_signal_to_csv(...)` fonksiyonlarını yaz.

6. Gerekiyorsa `rules.py` içindeki kuralları, `docs/client-message.md` ve `docs/gosterge-ozet.md` ile tam hizalamak için küçük ayarlamalar yap, ancak **temel yapı korunmalı**.