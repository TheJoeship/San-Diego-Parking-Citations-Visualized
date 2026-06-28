from pathlib import Path
import pandas as pd
import re

DATA_DIR = Path("data")
csv_files = sorted(DATA_DIR.glob("*.csv"))
print(f"Found {len(csv_files)} files")

df = pd.read_csv(csv_files[0])    # read just the first one
print(df.shape)                   # (rows, columns)
print(df.columns.tolist())        # what are the columns called?
print(df.dtypes)                  # what type did pandas guess for each?
df.head()                         # first 5 rows

#Ensure all .csv files have matching schema
for f in csv_files:
    cols = pd.read_csv(f, nrows=0).columns.tolist()     # nrows=0 = read header only, instant
    print(f.name, "->", cols)

#Combine into one data frame
df = pd.concat(
    [pd.read_csv(f,dtype=str) for f in csv_files],
    ignore_index=True
)

##################################### Initial Interogation ###################################################
print(f"Total rows: {len(df):,}")

#Now df is 13 years worth of parking tickets, let's look at some data
df["date_issue"] = pd.to_datetime(df["date_issue"],errors="coerce")         #Coerced errors turn into NaT and date text strings are converted to datetime objs
df["vio_fine"] = pd.to_numeric(df["vio_fine"],errors="coerce")

print("Date Range:",df["date_issue"].min(), "to", df["date_issue"].max())
print("Rows that failed date parsing:", df["date_issue"].isna().sum())
print("Rows that failed fine parsing:", df["vio_fine"].isna().sum())
print("Number of null locations:",df["location"].isna().sum())

print("\nTop violation types:")
print(df["vio_desc"].value_counts().head(15))

print("\nTotal value of Fines:")
print(df["vio_fine"].sum())

print("\nBusiest locations:")
print(df["location"].value_counts().head(15))


print("\nUnique raw locations:", df["location"].nunique())

#What percentage of the entries are one-offs?
loc_counts = df["location"].value_counts()
print("\nLocations appearing only once:", (loc_counts == 1).sum())
print("Share of all unique locations that are one-offs:",
      f"{(loc_counts == 1).mean():.1%}")

#Look at a sample of the raw location strings
print("\nRandom sample of raw locations:")
for loc in df["location"].dropna().sample(30,random_state=1):
    print(repr(loc))



loc_counts = df["location"].value_counts()

# What fraction of TICKETS do the most common locations cover?
for top_n in [5_000, 10_000, 20_000, 50_000, 100_000]:
    coverage = loc_counts.head(top_n).sum() / len(df)
    print(f"Top {top_n:>7,} locations cover {coverage:.1%} of all tickets")


########################### Cleaning ###############################
def clean_address(addr):
    s = addr.upper().strip()
    s = s.replace(".", "")                              # "LN." -> "LN"

    # strip enforcement tags that are noise at block level
    s = re.sub(r'\b[NSEW]/ALLEY\b', '', s)              # E/ALLEY, W/ALLEY
    s = re.sub(r'\b[NSEW]CL\b', '', s)                  # ECL, SCL (curb line)
    s = re.sub(r'\b[NSEW]/[OLC]\b', '', s)              # N/O, W/L, C/L family
    s = re.sub(r'\b(NORTH|SOUTH|EAST|WEST)BOUND\b', '', s)  # EASTBOUND

    # normalize the highest-impact street-type suffixes
    s = re.sub(r'\bAVENUE\b', 'AV', s)
    s = re.sub(r'\bAVE\b', 'AV', s)
    s = re.sub(r'\bSTREET\b', 'ST', s)
    s = re.sub(r'\bBOULEVARD\b', 'BL', s)
    s = re.sub(r'\bBLVD\b', 'BL', s)

    s = re.sub(r'\s+', ' ', s).strip()                 # collapse whitespace
    return s

#Clean UNIQUE strings only, then map back onto all rows
unique_locs = df["location"].dropna().unique()
cleaned_map = {loc: clean_address(loc) for loc in unique_locs}      #Dict w/ form: raw_string: cleaned_string
df["clean_location"] = df["location"].map(cleaned_map)              #New clean_location column

print("Unique raw locations:    ", df["location"].nunique())
print("Unique cleaned locations:", df["clean_location"].nunique())

#Checking what changed, check the dict, verify proper cleaning
changed = [(k, v) for k, v in cleaned_map.items() if k != v]
print(f"{len(changed):,} of {len(cleaned_map):,} unique strings changed\n")

import random
for k, v in random.sample(changed, 20):
    print(f"{k!r:45} -> {v!r}")