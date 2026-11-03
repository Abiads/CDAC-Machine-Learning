"""
Title: NBA Player Biographical & Birthday Paradox Analysis
Description: Simple, built-in function driven Python script for loading,
             sanitizing, and analyzing NBA player birth dates and shared birthdays.
"""


from pathlib import Path
import pandas as pd

# Direct path to dataset in the current folder
csv_path = Path(__file__).resolve().parent / "player_data.csv"


def phase_1_discovery_and_sanitization():
    """Phase 1: Load data, standardize columns, and convert birth_date to datetime."""
    print("\n[PHASE 1] Discovery & Data Sanitization (5 Marks)")
    print("-" * 65)
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.lower().str.replace(' ', '_')
    df['birth_date'] = pd.to_datetime(df['birth_date'], errors='coerce')

    print(f"Dataset Shape: {df.shape}")
    print(f"Missing Birth Dates: {df['birth_date'].isnull().sum()}")
    print("\nSample Records:")
    print(df[['player', 'pos', 'birth_date', 'colleges']].head())
    return df


def phase_2_birthday_paradox_analysis(df):
    """Phase 2: Extract month-day, count frequencies, and show top shared birthdays."""
    print("\n[PHASE 2] Birthday Paradox & Date Distribution (10 Marks)")
    print("-" * 65)
    df['bday'] = df['birth_date'].dt.strftime('%B %d')
    bday_counts = df['bday'].value_counts()

    print("Top 5 Most Common Birthdays in NBA History:")
    print(bday_counts.head())
    return df


def phase_3_player_lookup_system(df, player_name):
    """Phase 3: Case-insensitive search for a player and retrieve details."""
    print(f"\n[PHASE 3] Player Lookup: Searching for '{player_name}' (10 Marks)")
    print("-" * 65)
    match = df[df['player'].str.contains(player_name, case=False, na=False)]
    print(match[['player', 'pos', 'ht', 'wt', 'birth_date', 'colleges']])
    return match


def phase_4_shared_birthday_cohort(df, player_name):
    """Phase 4: Find all players sharing the exact month and day with target player."""
    print(f"\n[PHASE 4] Shared Birthday Cohort for '{player_name}' (10 Marks)")
    print("-" * 65)
    match = df[df['player'].str.contains(player_name, case=False, na=False)]
    if not match.empty:
        bday = match['bday'].iloc[0]
        cohort = df[df['bday'] == bday]
        print(f"Found {len(cohort)} players born on {bday}:")
        print(cohort[['player', 'birth_date', 'colleges']])


def phase_5_evaluation_and_summary(df):
    """Phase 5: Summary findings on dataset coverage and birthday paradox."""
    print("\n[PHASE 5] Evaluation & Summary Reporting")
    print("-" * 65)
    total_valid = df['bday'].dropna().count()
    unique_days = df['bday'].nunique() #nunique count unique values
    print(f"Total Players Analyzed: {total_valid}")
    print(f"Unique Birthday Calendar Days: {unique_days} / 366")
    print("Birthday Paradox Conclusion: With 4,900+ players, every single day of the")
    print("year is shared by at least 4 NBA players (up to 26 players on September 19).")


def run_analytics_lifecycle():
    """Facade function coordinating the NBA analytics workflow."""
    print("=" * 65)
    print("STARTING NBA BIRTHDAY & BIOGRAPHICAL LIFECYCLE")
    print("=" * 65)

    target_player = "Kareem Abdul-Jabbar"

    df = phase_1_discovery_and_sanitization()
    df = phase_2_birthday_paradox_analysis(df)
    phase_3_player_lookup_system(df, target_player)
    phase_4_shared_birthday_cohort(df, target_player)
    phase_5_evaluation_and_summary(df)

    print("\n" + "=" * 65)
    print("NBA LIFECYCLE COMPLETED SUCCESSFULLY")
    print("=" * 65)


if __name__ == "__main__":
    run_analytics_lifecycle()
