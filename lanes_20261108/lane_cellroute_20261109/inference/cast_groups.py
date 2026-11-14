import json, collections
b = json.load(open("experiments/lane_cellroute_20261109/out/budget_fp16.json"))
F = b["files"]
print("file-object:", F.get("<file object>"))
g = collections.defaultdict(lambda: [0, 0.0])
for f, v in F.items():
    c = v.get("cast")
    if not c: continue
    k = "/".join(f.split("/")[:3])
    g[k][0] += 1; g[k][1] += c[1] / 1e9
for k, v in sorted(g.items(), key=lambda x: -x[1][1]): print("%-75s %4d %7.2f" % (k, v[0], v[1]))
