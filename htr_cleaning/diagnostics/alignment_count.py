from collections import Counter
import json

issues = json.load(open("logs/meta/issues_index.json"))

c = Counter(i["tag"] for i in issues)

print(c)