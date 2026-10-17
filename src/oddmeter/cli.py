"""Command line interface for OddMeter."""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

from .demo import make_demo, write_demo
from .detector import detect


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="oddmeter", description="Find unusual rows in a CSV with a local Isolation Forest.")
    sub = p.add_subparsers(dest="command", required=True)
    demo = sub.add_parser("demo", help="generate sample telemetry and run anomaly detection")
    demo.add_argument("--output", default="oddmeter-report.csv", help="report CSV path (default: oddmeter-report.csv)")
    demo.add_argument("--seed", type=int, default=7, help="reproducible demo seed")
    analyze = sub.add_parser("analyze", help="analyze a CSV file")
    analyze.add_argument("csv", help="input CSV file")
    analyze.add_argument("--features", nargs="+", help="numeric columns; defaults to all numeric columns")
    analyze.add_argument("--contamination", type=float, default=0.05, help="expected outlier fraction, in (0, 0.5]")
    analyze.add_argument("--output", default="oddmeter-report.csv", help="report CSV path")
    analyze.add_argument("--seed", type=int, default=42, help="random seed")
    analyze.add_argument("--demo-label", help="optional label column for demo-only precision/recall")
    return p


def read_csv(path: str) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError("The input CSV needs a header row.")
        rows = list(reader)
    if not rows:
        raise ValueError("The input CSV has a header but no data rows.")
    return rows


def numeric_features(rows: list[dict[str, object]], columns: list[str]) -> list[str]:
    features = []
    for name in columns:
        values = [row.get(name) for row in rows]
        parsed = []
        for value in values:
            if value is None or (isinstance(value, str) and not value.strip()):
                continue
            try:
                parsed.append(float(value))
            except (ValueError, TypeError):
                break
        else:
            if parsed:
                features.append(name)
    return features


def write_report(path: str, rows: list[dict[str, object]], features: list[str], result) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fields = ["row_number", "is_anomaly", "anomaly_score", "main_signal", *rows[0].keys()]
    with target.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for i, row in enumerate(rows):
            writer.writerow({
                "row_number": i + 1,
                "is_anomaly": bool(result.flags[i]),
                "anomaly_score": f"{result.scores[i]:.6f}",
                "main_signal": result.main_signals[i],
                **row,
            })


def run(args: argparse.Namespace) -> int:
    if args.command == "demo":
        rows = make_demo(seed=args.seed)
        features = ["cpu_pct", "memory_pct", "requests_per_min", "error_pct"]
        result = detect(rows, features, contamination=0.05, seed=args.seed)
        write_report(args.output, rows, features, result)
        truth = np.array([row["known_event"] == "yes" for row in rows])
        predicted = result.flags
        tp = int(np.sum(truth & predicted))
        precision = tp / max(1, int(predicted.sum()))
        recall = tp / max(1, int(truth.sum()))
        print(f"OddMeter demo · {len(rows)} synthetic telemetry rows")
        print(f"Flagged {int(predicted.sum())} rows · planted-event precision {precision:.0%} · recall {recall:.0%}")
        print(f"Report: {Path(args.output).resolve()}")
        print("Note: demo metrics are measured against the synthetic labels; real data has no labels by default.")
        return 0

    rows = read_csv(args.csv)
    columns = args.features or list(rows[0].keys())
    features = numeric_features(rows, columns)
    if not features:
        raise ValueError("No numeric features found. Use --features to choose numeric CSV columns.")
    result = detect(rows, features, contamination=args.contamination, seed=args.seed)
    write_report(args.output, rows, features, result)
    count = int(result.flags.sum())
    print(f"Analyzed {len(rows)} rows using {len(features)} feature(s): {', '.join(features)}")
    print(f"Flagged {count} row(s) ({count / len(rows):.1%}); report: {Path(args.output).resolve()}")
    ranked = np.argsort(result.scores)[:min(5, count)]
    if len(ranked):
        print("Most unusual rows (CSV data-row numbers):")
        for idx in ranked:
            signal = result.main_signals[idx] or "n/a"
            print(f"  row {idx + 1}: score {result.scores[idx]:.4f} · strongest median/IQR deviation: {signal}")
    if args.demo_label:
        if args.demo_label not in rows[0]:
            raise ValueError(f"Label column {args.demo_label!r} was not found.")
        truth = np.array([str(row.get(args.demo_label, "")).strip().lower() in {"1", "true", "yes", "anomaly", "outlier"} for row in rows])
        tp = int(np.sum(truth & result.flags))
        precision = tp / max(1, int(result.flags.sum()))
        recall = tp / max(1, int(truth.sum()))
        print(f"Optional labeled comparison · precision {precision:.1%} · recall {recall:.1%}")
    return 0


def main() -> int:
    args = parser().parse_args()
    try:
        return run(args)
    except (OSError, ValueError) as exc:
        print(f"oddmeter: error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
