import argparse
import logging
import re
import zipfile
import tempfile
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

logger = logging.getLogger(__name__)

ZIP_RE = re.compile(
    r"^(?P<symbol>[A-Z0-9_]+)-aggTrades-(?P<date>\d{4}-\d{2}-\d{2})\.zip$"
)

def setup_logging(level: str):
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

def parquet_exists(path: Path) -> bool:
    return path.exists() and path.stat().st_size > 0

def read_aggtrades_csv(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(
        csv_path,
        dtype={
            "agg_trade_id": "int64",
            "price": "float64",
            "quantity": "float64",
            "first_trade_id": "int64",
            "last_trade_id": "int64",
            "transact_time": "int64",
        },
    )
    df["is_buyer_maker"] = df["is_buyer_maker"].astype("bool")

    df["side"] = df["is_buyer_maker"].map({True: -1, False: 1}).astype("int8")
    df["signed_quantity"] = df["side"] * df["quantity"]

    df["volume"] = df["price"] * df["quantity"]
    df["signed_volume"] = df["side"] * df["volume"]

    df = df.sort_values("transact_time").reset_index(drop=True)
    return df[
        [
            "transact_time",
            "price",
            "quantity",
            "volume",
            "side",
            "signed_quantity",
            "signed_volume",
            "is_buyer_maker",
            "agg_trade_id",
            "first_trade_id",
            "last_trade_id",
        ]
    ]

def convert_zip_to_parquet(
        zip_path: Path,
        out_dir: Path,
        overwrite: bool = False,
):
    m = ZIP_RE.match(zip_path.name)
    if not m:
        raise ValueError(f"Unexpected zip path: {zip_path.name}")

    symbol = m.group("symbol")
    date = m.group("date")

    out_path = out_dir / f"{symbol}-aggTrades-{date}.parquet"

    if parquet_exists(out_path) and not overwrite:
        logger.info("SKIP %s (exists)", out_path)
        return

    out_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("DOWNLOADING  %s", out_path.name)
    with zipfile.ZipFile(zip_path) as zf:
        csv_files = [n for n in zf.namelist() if n.endswith(".csv")]
        if not csv_files:
            raise RuntimeError(f"No CSV in {zip_path}")
        csv_name = csv_files[0]

        with tempfile.TemporaryDirectory() as td:
            tmp_csv = Path(td) / csv_name
            with zf.open(csv_name) as src, open(tmp_csv, "wb") as dst:
                dst.write(src.read())

            df = read_aggtrades_csv(tmp_csv)


    table = pa.Table.from_pandas(df, preserve_index=False)
    tmp_out = out_path.with_suffix(".parquet.part")
    pq.write_table(table, tmp_out, compression="zstd")
    tmp_out.replace(out_path)

    logger.info("OK  %s (%.2f MB)", out_path.name, out_path.stat().st_size / 1024 / 1024)

@dataclass
class Args:
    zips_dir: Path
    out_dir: Path
    overwrite: bool
    log_level: str

def parse_args() -> Args:
    p = argparse.ArgumentParser("aggTrades zip -> trade-level parquet")
    p.add_argument("--zips-dir", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--log-level", default="INFO")
    ns = p.parse_args()

    return Args(
        zips_dir=ns.zips_dir,
        out_dir=ns.out_dir,
        overwrite=ns.overwrite,
        log_level=ns.log_level,
    )

def run(args: Args):
    zips = sorted(args.zips_dir.glob("*-aggTrades-*.zip"))

    if not zips:
        raise SystemExit("No zip files found")

    for z in zips:
        convert_zip_to_parquet(z, args.out_dir, overwrite=args.overwrite)

    logger.info("DONE | saved_to=%s",  args.out_dir)
if __name__ == "__main__":
    args = parse_args()
    setup_logging(args.log_level)
    run(args)