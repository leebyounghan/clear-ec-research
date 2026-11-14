"""Exact five-fold and folds 0+4 HEX of a chain and of the cell-route refit on it. usage: hex_f04.py CHAIN.npz CORR.npz"""
import sys, json
import numpy as np
sys.path.insert(0, '/workspace/clear-ec-research/experiments/meta_ceiling_20261103/source')
import best_pipeline_20261106 as BP
y, fold = BP.y[:, 2], BP.fold; ok = y > 0
b = np.load(sys.argv[1])['final_hex']; p = np.load(sys.argv[2])['P']
ape = lambda q, r: float(np.mean(np.abs(q[r & ok] - y[r & ok]) / y[r & ok]) * 100); f04 = (fold == 0) | (fold == 4); al = np.ones(len(y), bool)
print(json.dumps(dict(chain_5fold=round(ape(b, al), 5), chain_f04=round(ape(b, f04), 5), corr_5fold=round(ape(p, al), 5), corr_f04=round(ape(p, f04), 5))))
