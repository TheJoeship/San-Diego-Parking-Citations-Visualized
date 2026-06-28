"""
geocode.py  -- Census batch geocoder with per-batch checkpointing.

Run this ONCE, on its own, after your main script has written targets.csv.

  Input : targets.csv          (one column "address" of cleaned addresses)
  Output: geocoded_locations.csv  (address, lat, lon  -- matched rows only)

It chunks the addresses into <=10k batches (the Census limit), POSTs each to
the Census batch geocoder, and saves every chunk to geocode_cache/ the moment
it succeeds. If the run dies partway, just run it again -- completed chunks are
skipped, so you never re-do finished work or re-hit the API for them.

No API key needed; the Census geocoder is public.
"""
import io
import time
from pathlib import Path

import pandas as pd
import requests

# --- config ---
CENSUS_URL = "https://geocoding.geo.census.gov/geocoder/locations/addressbatch"
BENCHMARK = "Public_AR_Current"   # current address-range benchmark
CHUNK_SIZE = 10_000               # Census hard limit per upload
CITY, STATE = "SAN DIEGO", "CA"

TARGETS_FILE = Path("targets.csv")
OUTPUT_FILE = Path("geocoded_locations.csv")
CACHE_DIR = Path("geocode_cache")
CACHE_DIR.mkdir(exist_ok=True)

# Census response is a headerless CSV in this fixed column order:
RESULT_COLS = [
    "id", "input_address", "match", "match_type",
    "matched_address", "coords", "tiger_line_id", "side",
]


def build_census_chunk(locations):
    """Turn a list of cleaned address strings into Census batch CSV text.
    Strict format: id,street,city,state,zip  -- NO header row.
    We use the address itself as the id so results (which may come back in a
    different order) can be rejoined on it later."""
    buf = io.StringIO()
    for loc in locations:
        safe = str(loc).replace('"', "")          # strip quotes that'd break CSV
        buf.write(f'"{safe}","{safe}","{CITY}","{STATE}",""\n')
    return buf.getvalue()


def geocode_chunk(locations, retries=3, timeout=180):
    """POST one chunk, return a parsed DataFrame. Retries with backoff."""
    csv_text = build_census_chunk(locations)
    files = {"addressFile": ("chunk.csv", csv_text, "text/csv")}
    data = {"benchmark": BENCHMARK}

    for attempt in range(1, retries + 1):
        try:
            resp = requests.post(CENSUS_URL, files=files, data=data, timeout=timeout)
            resp.raise_for_status()
            return pd.read_csv(
                io.StringIO(resp.text),
                header=None, names=RESULT_COLS, dtype=str,
            )
        except (requests.RequestException, pd.errors.ParserError) as e:
            print(f"    attempt {attempt}/{retries} failed: {e}")
            time.sleep(5 * attempt)               # back off a little each retry
    raise RuntimeError("chunk failed after all retries")


def split_coords(df):
    """Census returns matched coords as 'lon,lat' (LONGITUDE FIRST).
    Split into numeric lat/lon columns; unmatched rows get NaN."""
    matched = df["match"] == "Match"
    lonlat = df.loc[matched, "coords"].str.split(",", expand=True)
    df["lon"] = pd.to_numeric(lonlat[0], errors="coerce")
    df["lat"] = pd.to_numeric(lonlat[1], errors="coerce")
    return df


def geocode_all(targets):
    """Geocode a list of cleaned addresses, chunked + checkpointed."""
    chunks = [targets[i:i + CHUNK_SIZE] for i in range(0, len(targets), CHUNK_SIZE)]
    print(f"{len(targets):,} addresses -> {len(chunks)} chunk(s)")

    results = []
    for n, chunk in enumerate(chunks):
        cache_file = CACHE_DIR / f"chunk_{n:03d}.csv"
        if cache_file.exists():                   # checkpoint: skip finished work
            print(f"chunk {n}: cached, skipping")
            results.append(pd.read_csv(cache_file, dtype=str))
            continue

        print(f"chunk {n}: geocoding {len(chunk):,} addresses...")
        t0 = time.time()
        df = geocode_chunk(chunk)
        df = split_coords(df)
        df.to_csv(cache_file, index=False)        # checkpoint immediately on success
        matched = (df["match"] == "Match").sum()
        print(f"  done in {time.time() - t0:.0f}s, {matched:,}/{len(df):,} matched")
        results.append(df)

    return pd.concat(results, ignore_index=True)


def main():
    if not TARGETS_FILE.exists():
        raise SystemExit(
            f"{TARGETS_FILE} not found. Run your main script first to create it "
            f'(pd.Series(targets, name="address").to_csv("targets.csv", index=False)).'
        )

    targets = pd.read_csv(TARGETS_FILE)["address"].dropna().astype(str).tolist()
    coords = geocode_all(targets)

    rate = (coords["match"] == "Match").mean()
    print(f"\nOverall match rate: {rate:.1%}")

    matched = coords[coords["match"] == "Match"].dropna(subset=["lat", "lon"])
    matched = matched.rename(columns={"id": "address"})
    matched[["address", "lat", "lon"]].to_csv(OUTPUT_FILE, index=False)
    print(f"Saved {len(matched):,} geocoded locations -> {OUTPUT_FILE}")

    # show a sample of failures so you can eyeball whether they're the
    # expected landmark / cross-street tail
    fails = coords[coords["match"] != "Match"]
    if len(fails):
        print(f"\n{len(fails):,} failures. Sample:")
        for addr in fails["id"].head(25):
            print(" ", addr)


if __name__ == "__main__":
    main()