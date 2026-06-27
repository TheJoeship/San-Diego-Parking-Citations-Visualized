from pathlib import Path
import pandas as pd

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

print(f"Total rows: {len(df):,}")

#Now df is 13 years worth of parking tickets, let's look at some data
df["date_issue"] = pd.to_datetime(df["date_issue"],errors="coerce")         #Coerced errors turn into NaT and date text strings are converted to datetime objs
df["vio_fine"] = pd.to_numeric(df["vio_fine"],errors="coerce")

print("Date Range:",df["date_issue"].min(), "to", df["date_issue"].max())
print("Rows that failed date parsing:", df["date_issue"].isna().sum())
print("Rows that failed fine parsing:", df["vio_fine"].isna().sum())

print("\nTop violation types:")
print(df["vio_desc"].value_counts().head(15))

print("\nTotal value of Fines:")
print(df["vio_fine"].sum())

print("\nBusiest locations:")
print(df["location"].value_counts().head(15))


print("Unique raw locations:", df["location"].nunique())