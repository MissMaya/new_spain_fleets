import json
from pathlib import Path
from collections import defaultdict

data = json.load(open("logs/meta/alignment_diagnostics.json"))

counts = defaultdict(int)

for d in data:
    counts[d["flag"]] += 1

print(counts)