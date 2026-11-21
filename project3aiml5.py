import numpy as np
import pandas as pd
from pathlib import Path

DATA_DIR = Path(".")
TRAIN_FILE = DATA_DIR / "train.csv"
STORE_FILE = DATA_DIR / "store.csv"


def load_data():
    """Load the two Rossmann CSV files."""
    if not TRAIN_FILE.exists() or not STORE_FILE.exists():
        raise FileNotFoundError(
            "Please place both train.csv and store.csv in the same folder "
            "as this Python script."
        )

    train = pd.read_csv(TRAIN_FILE)
    store = pd.read_csv(STORE_FILE)

    required = {"Store", "Date", "Sales"}
    missing = required - set(train.columns)
    if missing:
        raise ValueError(f"train.csv is missing columns: {sorted(missing)}")
    if "Store" not in store.columns:
        raise ValueError("store.csv must contain a 'Store' column.")

    return train, store


def clean_data(train, store):
    """Clean, combine, sort, and create basic forecasting features."""
    df = train.copy()

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["Sales"] = pd.to_numeric(df["Sales"], errors="coerce")
    df = df.dropna(subset=["Store", "Date", "Sales"])

    
    df = df[df["Sales"] >= 0].copy()
    df["Store"] = pd.to_numeric(df["Store"], errors="coerce")
    df = df.dropna(subset=["Store"])
    df["Store"] = df["Store"].astype(int)

   
    df = (
        df.sort_values(["Store", "Date"])
          .drop_duplicates(subset=["Store", "Date"], keep="last")
          .reset_index(drop=True)
    )

    
    df = df.merge(store, on="Store", how="left", suffixes=("", "_store"))

   
    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month
    df["Day"] = df["Date"].dt.day
    df["DayOfWeek"] = df["Date"].dt.dayofweek + 1

    
    grouped = df.groupby("Store", sort=False)["Sales"]
    df["Sales_Lag_1"] = grouped.shift(1)
    df["Sales_Lag_7"] = grouped.shift(7)
    df["Rolling_Mean_7"] = (
        df.groupby("Store")["Sales"]
          .transform(lambda s: s.shift(1).rolling(7, min_periods=1).mean())
    )

    return df


def run_backtesting(df):
    """
    Run a simple chronological one-step naïve forecast for each store.
    At least 3 observations are needed per store to create a test split.
    """
    results = []

    for store_id, group in df.groupby("Store"):
        group = group.sort_values("Date").reset_index(drop=True)
        n = len(group)

        if n < 3:
            results.append({
                "Store": store_id,
                "Train_Rows": n,
                "Test_Rows": 0,
                "MAE": np.nan,
                "RMSE": np.nan,
                "WAPE_Percent": np.nan,
                "Status": "Not enough rows: need at least 3 dates per store"
            })
            continue

        split = max(2, int(n * 0.8))
        if split >= n:
            split = n - 1

        actual = group.loc[split:, "Sales"].to_numpy(dtype=float)
      
        predicted = group.loc[split - 1:n - 2, "Sales"].to_numpy(dtype=float)

        errors = actual - predicted
        mae = float(np.mean(np.abs(errors)))
        rmse = float(np.sqrt(np.mean(errors ** 2)))
        total_actual = float(np.sum(np.abs(actual)))
        wape = (
            float(np.sum(np.abs(errors)) / total_actual * 100)
            if total_actual != 0 else np.nan
        )

        results.append({
            "Store": store_id,
            "Train_Rows": split,
            "Test_Rows": len(actual),
            "MAE": round(mae, 2),
            "RMSE": round(rmse, 2),
            "WAPE_Percent": round(wape, 2) if not np.isnan(wape) else np.nan,
            "Status": "Completed"
        })

    return pd.DataFrame(results)


def main():
    print("Loading CSV files...")
    train, store = load_data()

    print("Cleaning data and creating features...")
    cleaned = clean_data(train, store)

    cleaned.to_csv("cleaned_daily_sales.csv", index=False)

    print("Running backtesting...")
    backtest = run_backtesting(cleaned)
    backtest.to_csv("backtesting_results.csv", index=False)

    print("\nProject completed.")
    print(f"Input sales rows: {len(train):,}")
    print(f"Cleaned sales rows: {len(cleaned):,}")
    print(f"Stores found: {cleaned['Store'].nunique():,}")
    print("\nBacktesting results:")
    print(backtest.to_string(index=False))
    print("\nSaved files:")
    print("- cleaned_daily_sales.csv")
    print("- backtesting_results.csv")

    if len(cleaned) < 20:
        print(
            "\nNote: This is a very small sample. Backtesting needs at least "
            "3 dated sales records for each store. Use the full dataset for "
            "meaningful forecasting results."
        )


if __name__ == "__main__":
    main()
