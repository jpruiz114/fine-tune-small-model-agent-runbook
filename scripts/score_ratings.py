"""Unblind a filled-in ratings.csv and report the mean human rating per system."""
import csv
import json
import statistics
import sys
from pathlib import Path


def main():
    results = Path(sys.argv[1] if len(sys.argv) > 1 else "results/latest")
    key = json.loads((results / "ratings_key.json").read_text())
    scores = {"baseline": [], "fine_tuned": []}
    missing = 0
    with open(results / "ratings.csv", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if not row["rating_1_to_10"].strip():
                missing += 1
                continue
            system = key[row["example_id"]][row["candidate"]]
            scores[system].append(float(row["rating_1_to_10"]))
    report = {s: {"n": len(v), "mean": round(statistics.mean(v), 2) if v else None} for s, v in scores.items()}
    report["unrated_rows"] = missing
    (results / "ratings_summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
