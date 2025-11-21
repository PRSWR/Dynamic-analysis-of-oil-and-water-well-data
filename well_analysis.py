"""
Lightweight dynamic analysis tool for oil and water well production data.

Pure-Python implementation with zero external dependencies. Provides:
- Synthetic data generation.
- Decline-curve estimates, water cut, and cumulative production.
- Text-based charts for quick visualization.
- Demo interface that runs end-to-end with a single command.
"""
import argparse
import csv
import json
import math
import random
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

random.seed(42)


@dataclass
class WellConfig:
    name: str
    initial_oil_rate: float  # barrels per day
    initial_water_rate: float  # barrels per day
    decline_rate: float  # fractional per day
    water_cut_growth: float  # fractional increase per day


def generate_synthetic_data(days: int = 365, wells: List[WellConfig] = None) -> List[Dict[str, object]]:
    if wells is None:
        wells = [
            WellConfig("Well-A", 120.0, 40.0, 0.0018, 0.0008),
            WellConfig("Well-B", 95.0, 25.0, 0.0012, 0.0011),
            WellConfig("Well-C", 60.0, 15.0, 0.0020, 0.0015),
        ]

    start_date = datetime(2022, 1, 1)
    records: List[Dict[str, object]] = []

    for day in range(days):
        current_date = start_date + timedelta(days=day)
        for well in wells:
            oil_rate = well.initial_oil_rate * math.exp(-well.decline_rate * day)
            water_rate = well.initial_water_rate * (1 + well.water_cut_growth * day)

            oil_rate += random.gauss(0, oil_rate * 0.05)
            water_rate += random.gauss(0, water_rate * 0.05)

            flowing_pressure = 1500 - 0.6 * day + random.gauss(0, 10)
            pump_frequency = 45 + 5 * math.sin(day / 30) + random.gauss(0, 0.5)

            records.append(
                {
                    "date": current_date.strftime("%Y-%m-%d"),
                    "well": well.name,
                    "oil_rate_bpd": max(oil_rate, 0.0),
                    "water_rate_bpd": max(water_rate, 0.0),
                    "flowing_pressure_psia": max(flowing_pressure, 0.0),
                    "pump_frequency_hz": max(pump_frequency, 0.0),
                }
            )

    records.sort(key=lambda r: (r["well"], r["date"]))
    return records


def write_csv(path: Path, records: Iterable[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    records = list(records)
    fieldnames = list(records[0].keys()) if records else []
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def load_csv(path: Path) -> List[Dict[str, object]]:
    with path.open() as f:
        reader = csv.DictReader(f)
        return [
            {
                **row,
                "oil_rate_bpd": float(row["oil_rate_bpd"]),
                "water_rate_bpd": float(row["water_rate_bpd"]),
                "flowing_pressure_psia": float(row["flowing_pressure_psia"]),
                "pump_frequency_hz": float(row["pump_frequency_hz"]),
            }
            for row in reader
        ]


def decline_curve_fit(dates: List[str], rates: List[float]) -> Tuple[float, float]:
    # Exponential decline: q = qi * exp(-D * t)
    date_objs = [datetime.strptime(d, "%Y-%m-%d") for d in dates]
    t = [(d - date_objs[0]).days for d in date_objs]
    positive = [(ti, r) for ti, r in zip(t, rates) if r > 0]
    if len(positive) < 2:
        qi = rates[0] if rates else 0.0
        return qi, 0.0

    t, r = zip(*positive)
    y = [math.log(val) for val in r]
    n = len(t)
    sum_t = sum(t)
    sum_y = sum(y)
    sum_ty = sum(ti * yi for ti, yi in zip(t, y))
    sum_tt = sum(ti * ti for ti in t)

    # Linear regression y = a + b*t -> decline D = -b, qi = exp(a)
    denom = n * sum_tt - sum_t * sum_t
    if denom == 0:
        return r[0], 0.0
    b = (n * sum_ty - sum_t * sum_y) / denom
    a = (sum_y - b * sum_t) / n
    qi = math.exp(a)
    decline = -b
    return qi, decline


def moving_average(series: List[float], window: int) -> List[float]:
    result = []
    for i in range(len(series)):
        start = max(0, i - window + 1)
        window_vals = series[start : i + 1]
        result.append(sum(window_vals) / len(window_vals))
    return result


def ascii_chart(title: str, dates: List[str], values: List[float], width: int = 60) -> str:
    if not values:
        return f"{title}: no data"
    min_v, max_v = min(values), max(values)
    span = max_v - min_v or 1
    step = max(1, len(values) // width)
    sampled = values[::step]
    bars = [int((v - min_v) / span * 8) for v in sampled]
    palette = "▁▂▃▄▅▆▇█"
    plot = "".join(palette[min(b, 7)] for b in bars)
    return f"{title}\n{dates[0]} {plot} {dates[-1]}\nmin={min_v:.1f}, max={max_v:.1f}"


def analyze_dataset(records: List[Dict[str, object]], outdir: Path) -> Dict[str, Dict[str, float]]:
    outdir.mkdir(parents=True, exist_ok=True)
    by_well: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for row in records:
        by_well[row["well"]].append(row)

    summary: Dict[str, Dict[str, float]] = {}

    for well, rows in by_well.items():
        dates = [r["date"] for r in rows]
        oil_rates = [r["oil_rate_bpd"] for r in rows]
        water_rates = [r["water_rate_bpd"] for r in rows]

        water_cut = [wr / (wr + orate) * 100 if (wr + orate) else 0 for wr, orate in zip(water_rates, oil_rates)]
        cum_oil, cum_water = [], []
        running_oil = running_water = 0.0
        for o, w in zip(oil_rates, water_rates):
            running_oil += o
            running_water += w
            cum_oil.append(running_oil)
            cum_water.append(running_water)

        oil_ma30 = moving_average(oil_rates, 30)
        qi, decline = decline_curve_fit(dates, oil_rates)

        enriched_rows = []
        for base, wc, co, cw, ma in zip(rows, water_cut, cum_oil, cum_water, oil_ma30):
            enriched = dict(base)
            enriched.update(
                {
                    "water_cut_pct": wc,
                    "cum_oil_bbl": co,
                    "cum_water_bbl": cw,
                    "oil_rate_ma30": ma,
                }
            )
            enriched_rows.append(enriched)

        write_csv(outdir / f"{well}_enriched.csv", enriched_rows)

        summary[well] = {
            "qi_bpd": qi,
            "decline_per_day": decline,
            "avg_water_cut_pct": sum(water_cut) / len(water_cut),
            "cum_oil_bbl": cum_oil[-1],
            "cum_water_bbl": cum_water[-1],
        }

        chart = "\n\n".join(
            [
                ascii_chart(f"{well} Oil Rate", dates, oil_rates),
                ascii_chart(f"{well} Water Rate", dates, water_rates),
                ascii_chart(f"{well} Water Cut %", dates, water_cut),
            ]
        )
        (outdir / f"{well}_charts.txt").write_text(chart)

    (outdir / "summary.json").write_text(json.dumps(summary, indent=2))
    write_csv(outdir / "summary.csv", [
        {"well": w, **vals} for w, vals in summary.items()
    ])
    return summary


def format_summary_table(summary: Dict[str, Dict[str, float]]) -> str:
    headers = [
        "Well",
        "Qi (bpd)",
        "Decline/day",
        "Avg WC%",
        "Cum Oil (bbl)",
        "Cum Water (bbl)",
    ]
    rows = []
    for well, vals in summary.items():
        rows.append(
            [
                well,
                f"{vals['qi_bpd']:.1f}",
                f"{vals['decline_per_day']:.5f}",
                f"{vals['avg_water_cut_pct']:.2f}",
                f"{vals['cum_oil_bbl']:.1f}",
                f"{vals['cum_water_bbl']:.1f}",
            ]
        )
    col_widths = [max(len(row[i]) for row in [headers] + rows) for i in range(len(headers))]
    lines = [
        " | ".join(val.ljust(col_widths[i]) for i, val in enumerate(headers)),
        "-+-".join("-" * col_widths[i] for i in range(len(headers))),
    ]
    for row in rows:
        lines.append(" | ".join(val.ljust(col_widths[i]) for i, val in enumerate(row)))
    return "\n".join(lines)


def ensure_dataset(path: Path, days: int) -> None:
    if path.exists():
        return
    data = generate_synthetic_data(days=days)
    write_csv(path, data)
    print(f"Synthetic dataset created at {path}")


def run_demo(data_path: Path, outdir: Path, days: int) -> None:
    ensure_dataset(data_path, days)
    records = load_csv(data_path)
    summary = analyze_dataset(records, outdir)

    print("\n=== Quick Summary ===")
    print(format_summary_table(summary))
    print(f"\nArtifacts saved in: {outdir.resolve()}")
    print("Type a well name (e.g., Well-A) to view its chart snippet, or use the commands below.")

    charts_cache = {well: (outdir / f"{well}_charts.txt").read_text() for well in summary}

    help_text = (
        "Commands:\n"
        "  [well name] - show charts for that well\n"
        "  s - show summary table\n"
        "  r - regenerate data and re-run analysis\n"
        "  q - quit\n"
    )

    while True:
        choice = input("demo> ").strip()
        if choice.lower() == "q":
            print("Exiting demo. Files remain in the data/ and results/ folders.")
            break
        if choice.lower() == "s":
            print(format_summary_table(summary))
            continue
        if choice.lower() == "r":
            data = generate_synthetic_data(days=days)
            write_csv(data_path, data)
            records = load_csv(data_path)
            summary = analyze_dataset(records, outdir)
            charts_cache = {well: (outdir / f"{well}_charts.txt").read_text() for well in summary}
            print("Data regenerated and analysis refreshed. New summary:")
            print(format_summary_table(summary))
            continue
        if choice in charts_cache:
            print(charts_cache[choice])
            continue

        print(help_text)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dynamic analysis for oil & water well data")
    subparsers = parser.add_subparsers(dest="command", required=True)

    gen = subparsers.add_parser("generate-data", help="Create a synthetic dataset")
    gen.add_argument("--output", type=Path, default=Path("data/well_data.csv"), help="Output CSV path")
    gen.add_argument("--days", type=int, default=365, help="Number of days to simulate")

    ana = subparsers.add_parser("analyze", help="Analyze a dataset")
    ana.add_argument("--input", type=Path, default=Path("data/well_data.csv"), help="Input CSV path")
    ana.add_argument("--outdir", type=Path, default=Path("results"), help="Output directory for reports")

    demo = subparsers.add_parser("demo", help="Run a one-command demo with interactive prompts")
    demo.add_argument("--input", type=Path, default=Path("data/well_data.csv"), help="Dataset path (created if missing)")
    demo.add_argument("--outdir", type=Path, default=Path("results"), help="Where to write charts and summaries")
    demo.add_argument("--days", type=int, default=180, help="Days of synthetic data to generate if needed")

    return parser.parse_args()


def build_zipapp(output: Path = Path("dist/well_analyzer.pyz")) -> None:
    """Create a tiny zipapp containing only this module.

    The resulting `.pyz` archive still requires Python 3.9+, but it behaves like a
    single-file entry point when executed with `python well_analyzer.pyz`.
    """

    output.parent.mkdir(parents=True, exist_ok=True)
    import shutil
    import tempfile
    import zipapp

    with tempfile.TemporaryDirectory() as tmpdir:
        temp_path = Path(tmpdir)
        shutil.copy(Path(__file__), temp_path / Path(__file__).name)

        zipapp.create_archive(
            source=str(temp_path),
            target=str(output),
            main="well_analysis:main",
            interpreter="/usr/bin/env python3",
        )

    print(f"Zipapp created at {output}. Run it with: python {output.name} demo")


def main() -> None:
    args = parse_args()

    if args.command == "generate-data":
        data = generate_synthetic_data(days=args.days)
        write_csv(args.output, data)
        print(f"Synthetic dataset created at {args.output}")
    elif args.command == "analyze":
        if not args.input.exists():
            raise FileNotFoundError(f"Input file not found: {args.input}")
        records = load_csv(args.input)
        summary = analyze_dataset(records, args.outdir)
        print(json.dumps(summary, indent=2))
        print(f"Analysis artifacts saved to {args.outdir}")
    elif args.command == "demo":
        run_demo(args.input, args.outdir, args.days)


if __name__ == "__main__":
    main()
