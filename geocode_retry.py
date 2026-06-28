"""
geocode_retry.py  -- second-pass rescue for addresses that failed the first run.

The first pass sent every address as city="SAN DIEGO". That makes the Census
geocoder miss neighborhoods that TIGER files under a different place name +
ZIP -- most notably La Jolla (92037), which showed up as a big cluster of
clean-looking failures.

This script re-geocodes ONLY the failures, trying two alternate strategies per
address, and appends any new matches to geocoded_locations.csv. It reuses the
chunking/checkpoint machinery from geocode.py.

Run AFTER geocode.py, once:  python geocode_retry.py
"""
import io
import time
from pathlib import Path

import pandas as pd
import requests

from geocode import (
    CENSUS_URL, BENCHMARK, RESULT_COLS, split_coords,
    OUTPUT_FILE, CACHE_DIR,
)

# Known La Jolla street markers -> these get the 92037 / "LA JOLLA" treatment.
LA_JOLLA_MARKERS = [
    "IVANHOE", "WALL ST", "SILVERADO", "HERSCHEL", "LA JOLLA", "PROSPECT",
    "GIRARD", "FAY AV", "DRAPER", "CAVE ST", "AVENIDA DE LA PLAYA",
    "EL PASEO GRANDE", "BIRD ROCK", "COAST BL", "TORREY PINES",
]

RETRY_CACHE = CACHE_DIR / "retry_chunk_{:03d}.csv"


def build_retry_chunk(locations):
    """Same strict format, but choose city/zip per-address.
    La Jolla streets -> 'LA JOLLA', 92037. Everything else -> blank city,
    letting Census resolve the place from the street+state itself."""
    buf = io.StringIO()
    for loc in locations:
        safe = str(loc).replace('"', "")
        if any(m in safe for m in LA_JOLLA_MARKERS):
            city, zp = "LA JOLLA", "92037"
        else:
            city, zp = "", ""        # blank city: let Census infer it
        buf.write(f'"{safe}","{safe}","{city}","CA","{zp}"\n')
    return buf.getvalue()


def geocode_retry_chunk(locations, retries=3, timeout=180):
    csv_text = build_retry_chunk(locations)
    files = {"addressFile": ("chunk.csv", csv_text, "text/csv")}
    data = {"benchmark": BENCHMARK}
    for attempt in range(1, retries + 1):
        try:
            resp = requests.post(CENSUS_URL, files=files, data=data, timeout=timeout)
            resp.raise_for_status()
            return pd.read_csv(io.StringIO(resp.text), header=None,
                               names=RESULT_COLS, dtype=str)
        except (requests.RequestException, pd.errors.ParserError) as e:
            print(f"    attempt {attempt}/{retries} failed: {e}")
            time.sleep(5 * attempt)
    raise RuntimeError("retry chunk failed after all retries")


def main():
    # Reconstruct the failure list: targets that are NOT in the matched output.
    targets = pd.read_csv("targets.csv")["address"].dropna().astype(str)
    matched = pd.read_csv(OUTPUT_FILE)["address"].astype(str)
    fails = sorted(set(targets) - set(matched))
    print(f"{len(fails):,} addresses to retry")

    if not fails:
        print("Nothing to retry.")
        return

    CHUNK = 10_000
    chunks = [fails[i:i + CHUNK] for i in range(0, len(fails), CHUNK)]
    results = []
    for n, chunk in enumerate(chunks):
        cache = Path(str(RETRY_CACHE).format(n))
        if cache.exists():
            print(f"retry chunk {n}: cached")
            results.append(pd.read_csv(cache, dtype=str))
            continue
        print(f"retry chunk {n}: geocoding {len(chunk):,}...")
        df = geocode_retry_chunk(chunk)
        df = split_coords(df)
        df.to_csv(cache, index=False)
        won = (df["match"] == "Match").sum()
        print(f"  recovered {won:,}/{len(df):,}")
        results.append(df)

    retry_df = pd.concat(results, ignore_index=True)
    new_matches = retry_df[retry_df["match"] == "Match"].dropna(subset=["lat", "lon"])
    new_matches = new_matches.rename(columns={"id": "address"})[["address", "lat", "lon"]]

    # Append the rescued rows to the existing output.
    combined = pd.concat([pd.read_csv(OUTPUT_FILE), new_matches], ignore_index=True)
    combined = combined.drop_duplicates(subset="address")
    combined.to_csv(OUTPUT_FILE, index=False)

    print(f"\nRescued {len(new_matches):,} more locations.")
    print(f"Total geocoded now: {len(combined):,}")


if __name__ == "__main__":
    main()