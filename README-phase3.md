# Binance USDT Signal Bot – Faz 3 (Genişleme, Şeffaflık ve Gelecek Geliştirmeler)

Bu doküman, Faz 1 ve Faz 2 başarıyla tamamlandıktan sonra **Faz 3**’te yapılabilecek geliştirmeleri tanımlar.  
Faz 3’ün iki ana odağı vardır:

1. Stratejiyi **daha yüksek timeframe’lere** ve opsiyonel gelişmiş kurallara genişletmek.
2. Kullanıcıya ve geliştiriciye yönelik **şeffaflık ve yönetilebilirlik** artırmak.

Temel referans yine [client-message.md](client-message.md)’dir; özellikle:

- “4h (optional for future upgrades)”
- Prefilter ve cooldown sisteminin agresif piyasa koşullarında davranışı
- Blok bazlı stratejinin gelecekte adaptif hale getirilebilmesi.

---

## 1. Multi-Timeframe Yapının Genişletilmesi (4h Kullanımı)

Kaynak: `client-message.md > 6. Multi-Timeframe Structure`

### 1.1. Hedef

- Halihazırda:
  - 15m → ana timeframe,
  - 1h → trend confirmation olarak kullanılıyor.
- Faz 3’te:
  - 4h timeframe’i **yüksek seviye yön filtresi** veya **risk filtresi** olarak devreye almak.

### 1.2. Olası Uygulama Önerileri

1. `data_fetcher.fetch_multi_timeframe_klines`:
   - 4h verisini de aktif olarak çekiyor olmalı (Faz 1’de zaten planlandı).

2. `analyzer.py`:
   - 4h kapanışlar üzerinden:
     - EMA20 / EMA50,
     - Basit bir trend metriği (ör. son N kapanışın yukarı/aşağı eğilimi) hesaplanabilir.
   - `rules.decide_signal_label`’a opsiyonel bir parametre olarak “4h trend ok” bilgisi taşınabilir:
     - Örn. sadece 4h uptrend ise `ULTRA_BUY`’a izin verme gibi.

3. `config.py`:
   - 4h ile ilgili opsiyonel parametreler:
     - `ENABLE_4H_FILTER = True/False`
     - `FOUR_H_MIN_TREND_STRENGTH`, vb.

Bu noktada Faz 3, 4h verisini **zorunlu** kılmak zorunda değil; opsiyonel bir “güvenlik filtresi” olarak tasarlamak en esnek yaklaşım olur.

---

## 2. Prefilter Katmanının Dinamikleştirilmesi

Kaynak: `client-message.md > 5. Prefilter Layer`

Prefilter halihazırda statik eşiklerle çalışıyor (Volume, Price, 24h Change, Cooldown). Faz 3’te:

### 2.1. Hedef

- Piyasa koşullarına göre:
  - Çok fazla aday varsa eşiği dinamik olarak sertleştiren,
  - Çok az aday varsa (örneğin ayı piyasası) eşiği bir miktar yumuşatan
bir prefilter sistemi.

### 2.2. Örnek Yaklaşım

1. `scan_market` içinde:
   - Prefilter öncesi toplam USDT çifti sayısını,
   - Prefilter sonrası kalan sembol sayısını ölç.
2. Eğer:
   - Kalan sembol sayısı, `MAX_SYMBOLS_PER_SCAN`in çok üzerindeyse:
     - Dinamik olarak:
       - `MIN_24H_QUOTE_VOLUME` eşiklerini artır,
       - veya 24h change aralığını daralt.
   - Kalan sembol sayısı çok azsa:
     - Volume eşiğini azıcık düşürmeyi düşün (örneğin %10–20 esnetme).
3. Bu dinamikler, `config` içinde opsiyonel flag’ler ile kontrol edilebilir:
   - `ENABLE_DYNAMIC_PREFILTER = True/False`
   - `DYNAMIC_PREFILTER_TOLERANCE` gibi.

---

## 3. Sinyal Yönetimi ve Cooldown Sisteminin Zenginleştirilmesi

Kaynak: `client-message.md > 5. Prefilter Layer > Cooldown System`

Faz 3’te:

- Cooldown sadece “aynı coin için X dakikada bir sinyal üretme”den çıkarılıp:

  - **Sinyal kalitesi** ve **başarı oranı** ile ilişkilendirilebilir.
  - (İleri seviye, şu an için sadece yapısal alanları açmak yeterli.)

### 3.1. Geliştirme Fikirleri

1. `signals_log.csv` üzerinden:
   - Geçmiş sinyaller, fiyat hareketleri ile offline analiz edilebilir.
   - Faz 3’te:
     - Basit bir “geri bildirim mekanizması” eklenebilir:
       - Örn. TP/SL hit olup olmadığına göre sembol bazlı cool-down veya “risk flag” ayarlaması.

2. API Seviyesinde:
   - Şimdilik sadece yerel state (`last_signal_times`) kullanılıyor.
   - İstenirse, sinyal geçmişini bir küçük SQLite veya JSON dosyası üzerinden persist etmek (sunucu restart’ında kaybolmaması için).

---

## 4. Kullanıcıya Şeffaf Raporlama (Telegram & Log Geliştirmeleri)

Client dokümanı, botun neye baktığını oldukça net yazıyor. Faz 3’te amaç:

- Kullanıcıya atılan her sinyalde, **o bloktaki durumun küçük bir özetini** göstermek.

### 4.1. Telegram Mesaj Formatı Genişletmesi

`telegram_bot.format_signal_message(signal)` içinde:

- Faz 2’de eklenen `details` yapıları kullanılarak:

  - Trend:
    - ADX, DI+–, EMA hizalanması, MACD hist, Momentum, AO, 1h trend.
  - Osc:
    - RSI (ve hangi bandta), StochK, CCI, Stoch RSI, Williams %R, UO.
  - Volume:
    - OBV trend (son 10 bar), volume spike faktörü, Bull/Bear power.
  - Price Action:
    - Hammer, strong green, EMA20 breakout, no collapse, min volume.

- Mesajı aşırı uzun yapmadan, kritik birkaç metriği göstermeyi hedefle:
  - Örneğin:
    ```text
    🔔 STRONG_BUY – BTCUSDT
    Price: 42,000 USDT (24h: +3.2%, Vol: 2.1B USDT)

    Trend (4): ADX 29, DI+>DI-, Price>EMA20>EMA50, 1h uptrend
    Osc (3): RSI 54 (healthy), StochK>50, UO>50
    Vol (3): OBV uptrend, Vol spike 1.8x, Bull>Bears
    PA (2): Hammer + EMA20 breakout, no dump in 96 bars
    ```

### 4.2. Log Geliştirmeleri

- CSV’ye:
  - Ek kolonlar eklenebilir (backward compatible olarak):
    - Örn. `adx`, `rsi`, `obv_trend`, `volume_spike_factor`, `ema_alignment_ok`, vb.
- Bu sayede gelecekte:
  - Offline backtest,
  - Strateji optimizasyonu
  kolaylaşır.

---

## 5. Yapılandırılabilirlik ve Modülerlik

Faz 3’te, özellikle **müşterinin istekleri değiştikçe** hızlı reaksiyon verebilmek için:

### 5.1. Kural Ağırlıklarını Konfigürasyona Taşıma

Şu anda `rules.py` içinde puanlar sabit (örn. MACD hist rising +1.5).  
Faz 3’te:

- Bu katsayılar `config.py` ya da ayrı bir `rules_config.py` içinde tutulabilir:
  - `TREND_ADX_STRONG_SCORE = 2`
  - `TREND_MACD_HIST_RISING_SCORE = 1.5`
  - vb.

Böylece:

- Kod değiştirmeden, sadece konfig değişikliği ile ince ayar yapılabilir.

### 5.2. Blok Bazlı Aç/Kapa

`config`’e:

- `ENABLE_TREND_BLOCK = True`
- `ENABLE_OSC_BLOCK = True`
- `ENABLE_VOLUME_BLOCK = True`
- `ENABLE_PRICE_ACTION_BLOCK = True`

gibi flag’ler eklenebilir.  
Bu sayede:

- Test ortamında örneğin sadece:
  - Trend + Volume blokları aktif,
  - Oscillator blokları devre dışı gibi senaryolar denenebilir.

---

## 6. Backtest CLI Notları

- Faz 3 testlerinde 15m ve 1h datasetleri artık ayrı dizinlerde tutuluyor; `src/backtest.py` çalıştırılırken `--data-dir-15m` ve `--data-dir-1h` parametrelerini birlikte ver.
- Çoklu TP/SL varyasyonları için `--num-cycles` artırılarak daha geniş kombinasyon uzayı taranabilir.
- Skor segmentlerini özelleştirmek için `--score-buckets` kullan; örn. `--score-buckets "7-8:mid,9-10,11+:top"`.
- Uzun batch çalışmaları için örnek komut:

```pwsh
python -m src.backtest --data-dir-15m data/precomputed_15m --data-dir-1h data/precomputed_1h --symbols ALL --strategies fut_safe,fut_aggressive --num-cycles 30 --score-buckets "8-9,10-11,12+" --results-dir results/faz3
```

- Özet dosyası (`summary.md`) artık 15m/1h dizinlerini ve cycle sayısını ayrı satırlarda raporlar; uzun raporlar için bu alanları arşivle.

## 7. Faz 3 Teslim Kriterleri

- 4h timeframe verisi çekiliyor ve opsiyonel filter/fonksiyonlar için kullanılabilecek durumda.
- Prefilter dinamik eşiğe uygun hale getirilebilecek altyapıya sahip (veya ilk versiyonu uygulanmış).
- Sinyal mesajları (Telegram + log) blok bazlı ayrıntılı özet sunabiliyor.
- Kuralların ağırlıkları ve bazı bloklar konfig ile yönetilebilir hale getirilmiş.
- Sistem uzun vadede:
  - Yeni timeframe eklemeye,
  - Yeni indikatör eklemeye,
  - Yeni blok eklemeye
  uygun modüler yapıda.

Faz 3 tamamlandığında, bot yalnızca sinyal üreten bir araç olmaktan çıkıp; **stratejisi şeffaf, yönetilebilir ve kolay optimize edilebilir** bir sinyal platformuna dönüşmüş olur.