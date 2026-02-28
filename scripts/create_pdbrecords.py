import argparse
import gzip
import json
import os

import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate gzipped JSON protein records from an input CSV file."
    )
    parser.add_argument("--data_file", required=True, help="Path to CSV file")
    parser.add_argument(
        "--out_file",
        required=True,
        help="Output path for gzipped JSON protein records",
    )
    return parser.parse_args()


def _as_clean_str(value):
    return value.strip() if isinstance(value, str) else value


def build_records(df: pd.DataFrame, data_file: str) -> dict:
    required = {"pdbpath", "sequence"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(
            f'Missing required column(s) in "{data_file}": {", ".join(sorted(missing))}'
        )

    records = {}
    conflicts = []

    for index, row in df.iterrows():
        row_num = index + 2  # account for header row
        pdbpath = _as_clean_str(row["pdbpath"])
        sequence = _as_clean_str(row["sequence"])

        if not pdbpath:
            raise ValueError(f'Empty "pdbpath" in row {row_num} of "{data_file}".')
        if not sequence:
            raise ValueError(f'Empty "sequence" in row {row_num} of "{data_file}".')

        key = os.path.basename(pdbpath)
        existing = records.get(key)
        if existing is not None and existing["seq"] != sequence:
            conflicts.append((row_num, key))
            continue

        records[key] = {"name": key, "seq": sequence}

    if conflicts:
        preview = ", ".join(
            [f'{key} (row {row_num})' for row_num, key in conflicts[:5]]
        )
        raise ValueError(
            "Found pdbpath basenames reused for different sequences. "
            f"Each unique sequence must have a unique pdbpath. Examples: {preview}"
        )

    return records


def main():
    args = parse_args()
    df = pd.read_csv(args.data_file)
    records = build_records(df, args.data_file)
    with gzip.open(args.out_file, "wt", encoding="utf-8") as handle:
        json.dump(records, handle)


if __name__ == "__main__":
    main()
