import numpy as np
import pandas as pd


def generate_dataset(output_file: str = "15_cinema_sales_1000.csv") -> pd.DataFrame:
    """Generate a 1,000-row synthetic cinema sales dataset (T00001 to T01000)."""
    np.random.seed(42)

    # 1. Create 985 unique Transaction_IDs spanning T00001 to T01000 + 15 duplicates = 1,000 rows
    middle_ids = np.random.choice(range(2, 1000), size=983, replace=False)
    unique_nums = np.sort(np.concatenate(([1], middle_ids, [1000])))
    n_unique = len(unique_nums)  # 985
    n_dups = 15

    tx_ids = [f"T{str(i).zfill(5)}" for i in unique_nums]

    # 2. Movies & genres
    movies_genres = [
        ("Starlight Odyssey", "Sci-Fi"),
        ("Midnight Chase", "Action"),
        ("Chasing Laughter", "Comedy"),
        ("Whispers in the Rain", "Drama"),
        ("Pixel Kingdom", "Animation"),
        ("Shadow Protocol", "Thriller"),
        ("Cosmic Horizon", "Sci-Fi"),
        ("Golden Hour", "Drama"),
        ("Turbo Drift", "Action"),
        ("The Secret Reef", "Animation"),
        ("Office Mayhem", "Comedy"),
        ("Echoes of Tomorrow", "Sci-Fi"),
    ]

    showtime_slots = ["Matinee", "Evening", "Late Night"]
    screen_formats = ["Standard 2D", "3D", "IMAX"]

    # 3. Show dates & weekend indicator
    start_date = pd.to_datetime("2025-01-01")
    end_date = pd.to_datetime("2026-11-15")
    days_range = (end_date - start_date).days

    random_days = np.random.randint(0, days_range + 1, size=n_unique)
    show_dates = start_date + pd.to_timedelta(random_days, unit="D")
    is_weekend = show_dates.dayofweek >= 5

    # 4. Categorical attributes
    movie_indices = np.random.choice(
        len(movies_genres),
        size=n_unique,
        p=[0.12, 0.11, 0.10, 0.08, 0.09, 0.08, 0.09, 0.06, 0.08, 0.07, 0.06, 0.06],
    )
    movie_titles = [movies_genres[i][0] for i in movie_indices]
    genres = [movies_genres[i][1] for i in movie_indices]

    slots = np.random.choice(showtime_slots, size=n_unique, p=[0.30, 0.45, 0.25])
    formats = np.random.choice(screen_formats, size=n_unique, p=[0.45, 0.30, 0.25])

    # 5. Ticket quantity (1-6 standard + 25 large group bookings of 8-15 for IQR outlier detection)
    base_qty = np.random.choice(
        [1, 2, 3, 4, 5, 6], size=n_unique, p=[0.20, 0.30, 0.20, 0.16, 0.08, 0.06]
    )
    outlier_idx = np.random.choice(n_unique, size=25, replace=False)
    base_qty[outlier_idx] = np.random.randint(8, 16, size=25)

    # 6. Ticket price by format + weekend adjustment
    ticket_prices = []
    for fmt, wknd in zip(formats, is_weekend):
        if fmt == "Standard 2D":
            price = np.random.uniform(160.0, 260.0)
        elif fmt == "3D":
            price = np.random.uniform(250.0, 360.0)
        else:  # IMAX
            price = np.random.uniform(350.0, 490.0)
        if wknd:
            price *= np.random.uniform(1.05, 1.15)
        ticket_prices.append(round(price, 2))

    # 7. Occupancy percentage by format, showtime slot, and weekend
    occupancies = []
    for fmt, slot, wknd in zip(formats, slots, is_weekend):
        base_occ = 45.0
        if fmt == "IMAX":
            base_occ += 15.0
        elif fmt == "3D":
            base_occ += 7.0
        if slot == "Evening":
            base_occ += 14.0
        elif slot == "Late Night":
            base_occ += 3.0
        if wknd:
            base_occ += 16.0
        occ = np.clip(np.random.normal(base_occ, 12.0), 18.0, 100.0)
        occupancies.append(round(occ, 1))

    # 8. Concession spend (with ~15% zeros and 28 missing values before duplication)
    concessions = []
    for qty, wknd in zip(base_qty, is_weekend):
        if np.random.rand() < 0.15:
            concessions.append(0.0)
        else:
            spend = qty * np.random.uniform(75.0, 165.0) * (1.12 if wknd else 1.0)
            concessions.append(round(spend, 2))

    concessions = np.array(concessions, dtype=float)
    nan_idx = np.random.choice(n_unique, size=28, replace=False)
    concessions[nan_idx] = np.nan

    # 9. Assemble DataFrame, inject 15 duplicate rows, and sort by Transaction_ID
    df_unique = pd.DataFrame(
        {
            "Transaction_ID": tx_ids,
            "Show_Date": show_dates.strftime("%Y-%m-%d"),
            "Movie_Title": movie_titles,
            "Genre": genres,
            "Showtime_Slot": slots,
            "Screen_Format": formats,
            "Tickets_Quantity": base_qty,
            "Ticket_Price": ticket_prices,
            "Occupancy_Percentage": occupancies,
            "Concession_Spend": concessions,
        }
    )

    dup_rows = df_unique.iloc[1:-1].sample(n=n_dups, random_state=42)
    df_1000 = (
        pd.concat([df_unique, dup_rows], ignore_index=True)
        .sort_values("Transaction_ID")
        .reset_index(drop=True)
    )

    df_1000.to_csv(output_file, index=False)
    print(f"Generated '{output_file}' with shape {df_1000.shape}")
    print(
        f"ID Range: {df_1000['Transaction_ID'].iloc[0]} to {df_1000['Transaction_ID'].iloc[-1]}"
    )
    print(f"Missing Concession_Spend values: {df_1000['Concession_Spend'].isnull().sum()}")
    print(
        f"Duplicate Transaction_ID rows: {df_1000.duplicated(subset=['Transaction_ID']).sum()}"
    )
    return df_1000


def clean_and_analyze(
    df: pd.DataFrame, cleaned_output: str = "15_cinema_sales_1000_cleaned.csv"
) -> pd.DataFrame:
    """Clean duplicates & missing values, engineer features, and run Project 15 tasks."""
    # Task 2: Remove duplicate Transaction_ID records
    df_clean = df.drop_duplicates(subset=["Transaction_ID"], keep="first").copy()

    # Standardize strings & Task 4: Convert Show_Date to datetime
    for col in ["Transaction_ID", "Movie_Title", "Genre", "Showtime_Slot", "Screen_Format"]:
        df_clean[col] = df_clean[col].astype(str).str.strip()

    df_clean["Show_Date"] = pd.to_datetime(df_clean["Show_Date"], errors="coerce")
    df_clean["Day_Type"] = np.where(
        df_clean["Show_Date"].dt.dayofweek >= 5, "Weekend", "Weekday"
    )

    # Task 3: Impute missing Concession_Spend using median grouped by Tickets_Quantity
    median_by_qty = df_clean.groupby("Tickets_Quantity")["Concession_Spend"].transform(
        "median"
    )
    overall_median = df_clean["Concession_Spend"].median()
    df_clean["Concession_Spend"] = (
        df_clean["Concession_Spend"].fillna(median_by_qty).fillna(overall_median).round(2)
    )

    # Task 5: Revenue calculations
    df_clean["Ticket_Revenue"] = (
        df_clean["Tickets_Quantity"] * df_clean["Ticket_Price"]
    ).round(2)
    df_clean["Total_Revenue"] = (
        df_clean["Ticket_Revenue"] + df_clean["Concession_Spend"]
    ).round(2)

    # Task 14: Ticket-quantity bands
    df_clean["Group_Band"] = pd.cut(
        df_clean["Tickets_Quantity"],
        bins=[0, 1, 2, 5, np.inf],
        labels=["Solo", "Couple", "Small Group", "Large Group"],
    )

    # Save cleaned dataset
    df_clean.to_csv(cleaned_output, index=False)
    print(f"\nSaved cleaned dataset to '{cleaned_output}' with shape {df_clean.shape}")

    # Key summary metrics (Tasks 6 - 13)
    print("\n--- TOP 10 MOVIES BY TICKET REVENUE ---")
    top_movies = (
        df_clean.groupby("Movie_Title", as_index=False)
        .agg(Total_Tickets=("Tickets_Quantity", "sum"), Ticket_Revenue=("Ticket_Revenue", "sum"))
        .sort_values("Ticket_Revenue", ascending=False)
        .head(10)
    )
    print(top_movies.to_string(index=False))

    print("\n--- SCREEN FORMAT OCCUPANCY ---")
    print(
        df_clean.groupby("Screen_Format")["Occupancy_Percentage"]
        .mean()
        .round(2)
        .to_string()
    )

    print("\n--- WEEKDAY VS WEEKEND ---")
    print(
        df_clean.groupby("Day_Type")[
            ["Tickets_Quantity", "Ticket_Revenue", "Concession_Spend"]
        ]
        .mean()
        .round(2)
        .to_string()
    )

    attach_rate = (df_clean["Concession_Spend"] > 0).mean() * 100
    print(f"\nConcession Attach Rate: {attach_rate:.2f}%")

    spend_arr = df_clean["Total_Revenue"].to_numpy()
    print(
        f"Total Transaction Spend -> Mean: {np.mean(spend_arr):.2f}, "
        f"Median: {np.median(spend_arr):.2f}, Std Dev: {np.std(spend_arr, ddof=1):.2f}"
    )

    q1 = df_clean["Tickets_Quantity"].quantile(0.25)
    q3 = df_clean["Tickets_Quantity"].quantile(0.75)
    iqr = q3 - q1
    upper_bound = q3 + 1.5 * iqr
    outliers = df_clean[df_clean["Tickets_Quantity"] > upper_bound]
    print(f"IQR Group Booking Outliers (> {upper_bound:.1f} tickets): {len(outliers)} records")

    return df_clean


if __name__ == "__main__":
    raw_df = generate_dataset("15_cinema_sales_1000.csv")
    cleaned_df = clean_and_analyze(raw_df, "15_cinema_sales_1000_cleaned.csv")

    # Auto-download files if running inside Google Colab
    try:
        from google.colab import files

        files.download("15_cinema_sales_1000.csv")
        files.download("15_cinema_sales_1000_cleaned.csv")
    except ImportError:
        pass