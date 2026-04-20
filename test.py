import ccxt
k = ccxt.kraken()
markets = k.load_markets()
btc = [s for s in markets if s.startswith('BTC/')]
print(sorted(btc))

