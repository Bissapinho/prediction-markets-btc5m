# Data notes

Facts checked by hand on live markets and on-chain before the analysis. The code refers to this file.

## Markets

- One market per 5-minute window, slug `btc-updown-5m-<unix start>`, start a multiple of 300 s. Listed through the
  public market API (`/events?series_id=10684`, pages of 100, cut into 6 h windows).
- The window start is the slug timestamp (`eventStartTime`). The `startDate` field is the creation date (~24 h earlier)
  and is never used.
- **Resolution rule.** Up wins if the 60-second TWAP of the Chainlink BTC/USD stream at the end of the window is at least
  the reference price, which is the TWAP of the previous window (`finalPrice(n) == priceToBeat(n + 1)` on every pair
  checked). Before ~2026-08-10 the markets resolved on the Chainlink *spot* price; agreement between the outcome and a
  Binance TWAP proxy is 85-88 % in May-July and 97-99 % from 2026-08-17. The study therefore starts on 2026-08-17.
- The outcome and the start/end prices come from the market API and serve as ground truth (0 disagreements with
  `finalPrice >= priceToBeat` on 10,000 markets).

## Trades

- Source: the public trade-history API in taker mode, one row per match (side, token and price of the taker).
  Up to 10,000 rows per request, so one request per market; the cap was never reached (max 9,075).
- Checked against the exchange contract's `OrderFilled` events on Polygon for 4 markets: 9,018 / 9,018 trades match,
  price difference < 1e-10.
- **Timestamps are block timestamps** (difference 0 s with the block). A block is mined ~2.25 s after the match is
  executed (see the README, "Measurement"). Every series built from trades is re-aligned by −2 s before it is compared
  with Binance.
- Taker fee, verified on-chain: `fee = 0.07 · p · (1 − p) · shares` USDC (exact on 92 % of fills; the rest are rounding
  or integrator fees). That is 1.75 c per share at p = 0.50 and 0.33 c at p = 0.95.
- 2026-09-19: no trade at all from 17:05 to 18:55 and from 19:20 to 20:10 UTC (an outage of the venue). Markets without
  any trade inside their window are excluded.

## Built tables

- `clean/trades`: one row per match. Key = blake2b(tx, token, wallet, side, size, price, ts), since the API has no log
  index. `seq` orders trades inside a market and only breaks ties within a second. Partition = UTC date of the market
  start.
- `features/btc_1s`: Binance BTCUSDT aggregated trades in 1-second bars [ts, ts + 1): OHLC, VWAP, volume, count.
  Empty seconds carry the previous close forward (`is_filled = true`).
- `features/market_1s`: 300 rows per market, t = 0..299, state **at the end** of second t. `p_up_implied` is the most
  recent of the last Up trade and 1 − the last Down trade; `up_age_s` / `down_age_s` give the age of the last trade.

## Simulated execution price

- There is no historical order book. The ask of a side at second t is the price of the last taker *buy* of that side
  within the previous 3 s.
- That price can be stale and below the true ask: in ~50 % of seconds, ask_up + ask_dn < 1 (the "pair arbitrage" A8 is
  this artefact). The Up and Down books mirror each other, so a buy of Down at q proves a bid for Up at 1 − q; the
  simulated ask is therefore floored: ask_up ← max(ask_up, 1 − ask_dn), and symmetrically (`PM_ASK=prudent`, default).
  Without this floor, a "price only" logistic model appeared to earn +5 c per share on the test set, with 95-98 % of its
  buys in those stale seconds.

## Binance vs Chainlink

- The Binance 60 s TWAP (BTCUSDT) sits ~0.06 % above Chainlink BTC/USD (level difference USDT vs USD). The fair-value
  model therefore works in Binance *returns* since the window start, not in levels. The Binance proxy reproduces the
  outcome in ~98 % of windows, but only ~88 % when Chainlink moves by less than 10 $.
