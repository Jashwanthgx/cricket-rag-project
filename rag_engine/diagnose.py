
import json
import os
from collections import defaultdict, Counter

DATA_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cricket_rag_data.jsonl")

style_bowlers = defaultdict(list)
style_innings = Counter()

with open(DATA_FILE) as f:
    for line in f:
        d = json.loads(line)
        if d["role"] != "bowler":
            continue
        style = d["metadata"].get("bowler_style", "Unknown")
        name = d["player_name"]
        style_innings[style] += 1
        style_bowlers[style].append(name)


pace_unique = sorted(set(style_bowlers["Pace"]))
spin_unique = sorted(set(style_bowlers["Spin"]))
unknown_unique = sorted(set(style_bowlers["Unknown"]))

print(f"\nUnique PACE bowlers ({len(pace_unique)}):")
for n in pace_unique[:80]:
    print(f"  {n}")

print(f"\nUnique SPIN bowlers ({len(spin_unique)}):")
for n in spin_unique[:80]:
    print(f"  {n}")

print(f"\nUnique UNKNOWN bowlers ({len(unknown_unique)}):")
for n in unknown_unique:
    print(f"  {n}")