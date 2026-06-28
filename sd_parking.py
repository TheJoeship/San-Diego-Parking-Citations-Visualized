from pathlib import Path
import pandas as pd
import re
import random

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

# Spelled-out ordinals -> numeric, so "THIRD AV" merges with "3RD AV".
# San Diego's numbered avenues are a known finite set, so this is safe.
ORDINALS = {
    "FIRST": "1ST", "SECOND": "2ND", "THIRD": "3RD", "FOURTH": "4TH",
    "FIFTH": "5TH", "SIXTH": "6TH", "SEVENTH": "7TH", "EIGHTH": "8TH",
    "NINTH": "9TH", "TENTH": "10TH", "ELEVENTH": "11TH", "TWELFTH": "12TH",
    "THIRTEENTH": "13TH", "FOURTEENTH": "14TH", "FIFTEENTH": "15TH",
    "SIXTEENTH": "16TH", "SEVENTEENTH": "17TH", "EIGHTEENTH": "18TH",
    "NINETEENTH": "19TH", "TWENTIETH": "20TH",
}

def clean_address(addr):
    s = addr.upper().strip()
    s = s.replace(".", " ")                                 # "N.MISSION" -> "N MISSION" (space, not nothing)

    # --- Slash-anchored position tags (CURB/SIDE/SID/LOT with a slash) ---
    # The mandatory "/" protects landmark words like "SKI BEACH LOT".
    # CURB\s*LINE listed FIRST so "CURB LINE" matches fully (not just "CURB").
    s = re.sub(
        r'[NSEW]?\s*/\s*[NSEW]?\s*(CURB\s*LINE|CURBLINE|CURB|SIDE|SID|LOT)\b(\s+(LN|LANE|CTR|CENTER))?',
        ' ', s,
    )

    # --- Spelled-out position tags: "EAST CURB LINE", "S SIDE", "W CURB" ---
    # Anchored on a LEADING direction (word or single letter) so bare landmark
    # words ("SKI BEACH LOT", "716 W UPAS ST") stay safe. CURB LINE first again.
    s = re.sub(
        r'\b(NORTH|SOUTH|EAST|WEST|N|S|E|W)\s+(CURB\s*LINE|CURB|SIDE|SID)\b',
        ' ', s,
    )

    # --- Curb-line / directional / position tags (abbreviated forms) ---
    s = re.sub(r'\b[NSEW]CL\b/?', '', s)                   # ECL, SCL, WCL, NCL
    s = re.sub(r'\b[NSEW]/ALLEY\b', '', s)                 # E/ALLEY, W/ALLEY
    s = re.sub(r'\b[NSEW]/A\b', '', s)                     # E/A, W/A (abbreviated alley)
    s = re.sub(r'\b[NSEW]/[OLC]\b', '', s)                 # N/O, W/L, C/L family
    s = re.sub(r'\b(NORTH|SOUTH|EAST|WEST)BOUND\b', '', s) # EASTBOUND etc.
    s = re.sub(r'\bIFO\b', '', s)                          # "in front of" - landmark noise

    # --- Spelled-out ordinals -> numeric ---
    for word, num in ORDINALS.items():
        s = re.sub(rf'\b{word}\b', num, s)

    # --- Street-type suffix normalization ---
    s = re.sub(r'\bAVENUE\b', 'AV', s)
    s = re.sub(r'\bAVE\b', 'AV', s)
    s = re.sub(r'\bSTREET\b', 'ST', s)
    s = re.sub(r'\b(BOULEVARD|BLVD)\b', 'BL', s)
    s = re.sub(r'\bDRIVE\b', 'DR', s)
    s = re.sub(r'\bROAD\b', 'RD', s)
    s = re.sub(r'\bCOURT\b', 'CT', s)
    s = re.sub(r'\bLANE\b', 'LN', s)
    s = re.sub(r'\bPLACE\b', 'PL', s)
    s = re.sub(r'\bTERRACE\b', 'TER', s)
    s = re.sub(r'\b(PARKWAY|PKWY)\b', 'PKWY', s)
    s = re.sub(r'\bCIRCLE\b', 'CIR', s)

    # --- Final cleanup (order matters) ---
    s = re.sub(r'\s*/\s*', ' ', s)                                              # orphaned slashes -> space
    s = re.sub(r'\s+', ' ', s)                                                  # normalize whitespace FIRST
    s = re.sub(r'\b(AV|ST|BL|DR|RD|CT|LN|PL|TER|CIR|PKWY)( \1\b)+', r' \1', s)  # collapse doubled suffixes
    s = re.sub(r'\s+', ' ', s).strip()                                          # final tidy + strip
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

#Random sample of 20 changes for analysis 
for k, v in random.sample(changed, 20):
    print(f"{k!r:45} -> {v!r}")

# Look specifically at strings still containing slashes or lone letters
hits = [(k, v) for k, v in changed if '/' in v or re.search(r'\b[NSEW]\b', v)]
print(f"{len(hits):,} strings still contain a slash or lone direction letter\n")

for k, v in hits[:15]:          # only print the first 15
    print(f"{k!r:45} -> {v!r}")


#Working on regex parsing, check occurences of spelled out numbers beyond 20 (i.e "thirtieth")
"""
for word in ["TWENTY", "THIRTIETH", "FORTIETH", "FIFTIETH", "SIXTIETH", "SEVENTIETH"]:
    n = df["location"].str.contains(rf'\b{word}', case=False, na=False).sum()
    print(f"{word:12} appears in {n:,} rows")
"""

#Now let's recheck our coverage and uniqueness after cleanup!
loc_counts = df["clean_location"].value_counts()

# What fraction of TICKETS do the most common locations cover?
for top_n in [5_000, 10_000, 20_000, 50_000, 100_000]:
    coverage = loc_counts.head(top_n).sum() / len(df)
    print(f"Top {top_n:>7,} clean locations cover {coverage:.1%} of all tickets")


# Rank cleaned locations by ticket count
loc_counts = df["clean_location"].value_counts()

# Take the head that covers ~80%+ of tickets
TOP_N = 50_000
geocode_targets = loc_counts.head(TOP_N).index.tolist()
print(f"Geocoding {TOP_N:,} locations covers "
      f"{loc_counts.head(TOP_N).sum() / len(df):.1%} of all tickets")