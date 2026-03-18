import json
from pathlib import Path

path = Path("logs/meta/alignment_diagnostics.json")

data = json.load(open(path))

drift = [d for d in data if d["flag"] == "ALIGNMENT_DRIFT"]

print("\nAlignment Drift Diagnostics\n")

print("Documents analysed:", len(data))
print("Documents flagged as drift:", len(drift))
print("Drift rate:", round(len(drift)/len(data)*100,2), "%\n")

print("Top drift documents:\n")

for d in sorted(drift, key=lambda x: x["total_s2"], reverse=True)[:20]:
    print(d["document_id"], "S2 issues:", d["total_s2"])