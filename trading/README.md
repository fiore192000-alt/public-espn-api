# Trading quantitativo: Freqtrade + modello XGBoost

Setup completo per far girare **sul tuo PC Windows** un bot che decide gli
ingressi con un modello di machine learning, testarlo sul passato e poi
lasciarlo lavorare in **dry-run** (prezzi veri, ordini simulati, zero denaro).

Niente API a pagamento, niente wallet, niente chiavi d'exchange. I dati storici
sono pubblici e gratuiti.

> Questa cartella è autonoma: non ha niente a che vedere con il resto del
> repository (`espn_service`, `nhl_service`), non ne condivide dipendenze né CI.

---

## 1. Prima di installare qualsiasi cosa

Apri **PowerShell** nella cartella `trading` ed esegui:

```powershell
.\scripts\00_check_environment.ps1
```

Fa esattamente i controlli che servono (`docker`, `python`, `git`, `nvidia-smi`),
ti dice cosa manca e cosa invece è inutile installare. In sintesi:

| Componente | Serve? | Nota |
|---|---|---|
| **Docker Desktop** | **Sì** (percorso consigliato) | È la via raccomandata da Freqtrade su Windows |
| Python 3.11+ | Solo per il percorso B (installazione nativa) | Dentro Docker c'è già |
| Git | Utile, non obbligatorio | Per aggiornare il codice |
| GPU NVIDIA | **No** | XGBoost qui gira su CPU, il dataset è piccolo |

Se PowerShell blocca gli script:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

---

## 2. I cinque passi

Ogni passo è uno script. Vanno eseguiti dalla cartella `trading`.

```powershell
.\scripts\01_build.ps1            # immagine Docker con freqtrade + xgboost
.\scripts\02_download_data.ps1    # candele storiche (gratis, pubbliche)
.\scripts\03_train.ps1            # allena il modello
.\scripts\04_backtest.ps1         # test sul periodo mai visto dal modello
.\scripts\05_dryrun.ps1           # bot live con ordini simulati
```

Tutti accettano parametri, per esempio:

```powershell
.\scripts\02_download_data.ps1 -Pairs "BTC/USDT","ETH/USDT" -Timeframes "1h" -Days 2000
.\scripts\03_train.ps1 -Timeframe 4h -Horizon 12 -TpAtr 2.5
.\scripts\04_backtest.ps1 -TimeRange 20240101-20250101
.\scripts\05_dryrun.ps1 -Follow
.\scripts\05_dryrun.ps1 -Stop
```

Su Linux, macOS o WSL gli script non servono: gli stessi comandi `docker compose`
che trovi dentro ognuno funzionano tali e quali.

---

## 3. Come ragiona il modello

### La domanda

Il modello non prova a indovinare il prezzo. Risponde a **una** domanda per
candela:

> Se entro all'apertura della prossima candela, il prezzo arriva a **+2 ATR**
> prima di scendere a **−1 ATR**?

È l'etichetta *triple-barrier*: barriera alta, barriera bassa, limite di tempo
(24 candele). Le barriere sono misurate in ATR, quindi significano la stessa
cosa in un mercato calmo e in uno isterico.

Perché non "il prezzo fra 24 ore sarà più alto"? Perché quella domanda non
corrisponde a nessuna operazione reale: un +3% raggiunto dopo un −8% è un
guadagno solo per chi non aveva uno stop.

### Gli input

45 feature, tutte calcolate **solo sul passato** (`user_data/ml/features.py`):

```
prezzo      rendimenti log 1/3/5/10/20/50 candele
trend       close/EMA20, EMA20/EMA50, EMA50/EMA200, pendenza 20 e 50
oscillatori RSI 7 e 14, MACD e istogramma (normalizzati sul prezzo), ADX, spread DI
volatilità  ATR%, ATR relativo alla media, dev. std. 20 e 50, rapporto fra le due
struttura   posizione e ampiezza Bollinger, distanza da massimi/minimi 20 e 50
candela     corpo, ombra superiore, ombra inferiore, range%
volume      z-score, rapporto sulla media, pressione volume-rendimento
mercato     rendimento BTC, volatilità BTC, trend BTC, correlazione 50, forza relativa
tempo       ora e giorno della settimana (codificati come seno/coseno)
```

L'output è una probabilità calibrata:

```
P(take-profit prima dello stop) = 0.71
```

"Calibrata" significa che 0.71 vuole davvero dire *71 volte su 100*: XGBoost da
solo produce punteggi ordinali, non probabilità, e vengono ricalibrati con una
regressione isotonica sui dati di validazione.

### Quando entra

Tre condizioni insieme (`MLProbStrategy.populate_entry_trend`):

1. `P >= soglia` — la soglia non è scelta a mano, la sceglie il trainer sui
   dati di validazione massimizzando il guadagno atteso per operazione;
2. la distanza dal take-profit vale il viaggio: `2 × ATR / prezzo >= 0.6%`,
   altrimenti commissioni e spread si mangiano tutto il margine;
3. c'è volume e l'ATR è definito.

L'uscita usa le stesse barriere dell'addestramento: take-profit a +2 ATR
(`custom_exit`), stop a −1 ATR (`custom_stoploss`), chiusura forzata dopo 24
candele. Backtest e metriche di training descrivono così la stessa cosa.

---

## 4. Perché i risultati non sono truccati

È la parte che decide se tutto il resto vale qualcosa. Quattro difese:

**Split temporale con purge.** Il modello si allena su tutto ciò che precede
`--train-end`, sceglie soglia e calibrazione su una coda di validazione che sta
comunque *prima* del test, e il periodo da `--test-start` in poi non viene mai
toccato. Le righe a cavallo di un confine vengono eliminate: l'etichetta della
riga *i* guarda fino a *i+24*, quindi le ultime 24 righe di ogni blocco
racconterebbero qualcosa che appartiene al blocco successivo.

```
        train                    valid      |  purge |        test
2019 ───────────────────── 2023-10    2023-12  |░░░░░░░░| 2024 ───── 2025
        fit del modello        soglia          |        |  mai visto
```

**Feature causali, verificate da un test.** `tests/test_features.py` calcola le
feature sulla serie intera e su una serie troncata: ogni riga in comune deve
risultare identica. Se qualcuno introduce uno `shift(-1)`, una media centrata o
una normalizzazione sull'intero campione, il test fallisce.

**Ingresso alla candela successiva.** L'etichetta assume l'entrata all'`open`
della candela *dopo* il segnale, mai al `close` della candela che lo genera —
che sarebbe un prezzo non ottenibile. È esattamente ciò che fa Freqtrade.

**Il controllo di Freqtrade.** `04_backtest.ps1` lancia anche
`lookahead-analysis`: rigira la strategia su storie accorciate e segnala se i
risultati cambiano, cioè se da qualche parte si sta leggendo il futuro.

**Un'ultima difesa che devi mettere tu:** non riallenare cambiando i parametri
finché il test non ti piace. A furia di guardarlo, il test diventa un secondo
set di validazione e smette di dire la verità.

---

## 5. Come leggere i numeri

Il training stampa due blocchi. Guarda **solo** la riga `test`:

```
Ranking quality
  valid   samples=41230  base_rate=0.3814  roc_auc=0.5732  ...
  test    samples=28904  base_rate=0.3556  roc_auc=0.5418  ...

Signal economics (barrier trades, fees included, no slippage)
  test    threshold=0.6250  trades=1204  win_rate=0.4402  expectancy=0.0031
          total_return=3.7324  profit_factor=1.2140
```

Ordini di grandezza onesti su dati reali:

| Metrica | Sospetto | Plausibile | Buono |
|---|---|---|---|
| ROC AUC (test) | > 0.65 | 0.52 – 0.56 | 0.57 – 0.62 |
| Profit factor (backtest) | > 3 | 1.05 – 1.25 | 1.3 – 1.6 |
| Numero di trade | < 100 | 300+ | 1000+ |

Un AUC di 0.75 sui prezzi non è un buon modello: è un errore da cercare. Un AUC
di 0.54 con abbastanza operazioni può invece essere un edge reale.

Nel backtest guarda in quest'ordine: **max drawdown** (riusciresti a starci
dentro?), profit factor, numero di trade, e il `--breakdown month` — un
risultato che nasce tutto da due mesi buoni non è una strategia.

Le "signal economics" del training sono più ottimistiche del backtest: ignorano
slippage, slot occupati e operazioni sovrapposte. Quando i due numeri litigano,
ha ragione il backtest. E quando backtest e dry-run litigano, ha ragione il
dry-run.

---

## 6. Dry-run

```powershell
.\scripts\05_dryrun.ps1
docker compose logs -f
```

`config.dryrun.json` ha `dry_run: true` e le chiavi exchange vuote, e lo script
si rifiuta di partire se qualcuno le riempie. Il bot non *può* piazzare un
ordine reale.

Lascialo girare **settimane, non giorni**, poi confronta il risultato con un
backtest sullo stesso periodo. Se divergono molto, il problema è quasi sempre
liquidità, slippage o un ordine che non si riempie mai al prezzo previsto.

---

## 7. Percorso B: senza Docker

Se preferisci l'installazione nativa (Python 3.11+):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install freqtrade
pip install -r requirements-ml.txt

freqtrade download-data --config user_data/config.dryrun.json --exchange binance `
    --pairs BTC/USDT ETH/USDT SOL/USDT --timeframes 1h --days 2200 --data-format-ohlcv feather

cd user_data
python -m ml.train --pairs BTC/USDT ETH/USDT SOL/USDT --timeframe 1h
cd ..

freqtrade backtesting --config user_data/config.dryrun.json --strategy MLProbStrategy --timerange 20240101-
```

Su Windows l'installazione nativa di Freqtrade richiede i Build Tools di Visual
Studio per compilare alcune dipendenze; se ti blocchi lì, torna a Docker.

---

## 8. Test

```powershell
pip install -r requirements-ml.txt
pytest
```

28 test, nessuna dipendenza da Freqtrade (viene sostituito da uno stub minimo):
verificano che le feature non guardino il futuro, che le etichette usino il
prezzo giusto, che lo split sia ordinato e purgato, che il modello si salvi e si
ricarichi, e che senza modello la strategia resti ferma invece di crashare.

---

## 9. Mappa dei file

```
trading/
├── docker-compose.yml           servizio bot + build dell'immagine
├── Dockerfile                   freqtrade + xgboost/scikit-learn
├── requirements-ml.txt
├── scripts/                     i cinque passi, in PowerShell
└── user_data/
    ├── config.dryrun.json       dry_run: true, chiavi vuote
    ├── ml/
    │   ├── features.py          le feature (causali, condivise train/live)
    │   ├── labeling.py          etichette triple-barrier
    │   ├── dataset.py           lettura dati Freqtrade, split con purge
    │   ├── evaluate.py          calibrazione, scelta soglia, metriche
    │   ├── train.py             CLI di addestramento
    │   └── model_io.py          salvataggio/caricamento del modello
    ├── strategies/
    │   └── MLProbStrategy.py    la strategia Freqtrade
    ├── models/entry/            modello allenato (non versionato)
    └── data/                    candele scaricate (non versionate)
```

Il punto chiave dell'organizzazione: **la strategia importa lo stesso
`features.py` che usa il training**. Le feature non possono divergere fra
addestramento ed esecuzione, che è il modo più comune di far morire un modello
in produzione senza accorgersene.

---

## 10. Prossimi passi sensati

Nell'ordine, se il primo giro dà segnali di vita:

1. **Walk-forward**: riallenare ogni 3-6 mesi invece di un unico split, e
   verificare che l'edge regga in tutti i periodi.
2. **Più coppie**: 10-20 coppie danno molti più campioni e un modello meno
   legato al singolo asset.
3. **Position sizing**: dimensionare in base a `P` e all'ATR invece di uno stake
   fisso.
4. **Modello di uscita**: oggi le uscite sono regole fisse; possono diventare un
   secondo modello.
5. **Hyperopt** su `entry_threshold` e `min_edge_pct`, con la stessa disciplina
   sui periodi.

Un LLM locale (Ollama) non serve a niente in questa pipeline: non aggiunge nulla
a un problema di dati tabellari e strutturati. Ha senso semmai altrove — leggere
notizie, produrre report — ma non per decidere gli ingressi.

---

## Avvertenza

Software per ricerca ed esperimenti. Un backtest positivo non è una previsione,
la maggior parte delle strategie profittevoli sul passato non lo è sul futuro, e
il dry-run esiste proprio perché tra i due c'è di mezzo il mercato vero. Non
usare denaro reale finché non hai visto la strategia comportarsi come previsto
per un periodo lungo, e comunque solo denaro che puoi permetterti di perdere.
