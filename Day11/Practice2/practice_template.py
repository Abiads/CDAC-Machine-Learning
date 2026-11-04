"""
================================================================================
EXAM PRACTICE (COMPLETE SOLUTION): Day 11 - Practice 2
NEW PROBLEM STATEMENT (PS - Target Player Variation):
--------------------------------------------------------------------------------
The scouting department wants to examine the birthday cohort of another legendary 
superstar: "Michael Jordan".

YOUR TASKS:
1. Parse birth dates safely handling malformed strings without crashing.
2. Find the top 5 most common shared birthdays in NBA history.
3. Build a case-insensitive lookup for "Michael Jordan" and retrieve his record.
4. Extract Michael Jordan's birth month and day, then retrieve all OTHER NBA players 
   born on that same calendar day.
================================================================================
"""

from pathlib import Path
import pandas as pd

csv_path = Path(__file__).resolve().parent / "player_data.csv"


def phase_1_discovery_and_sanitization():
    """
    [Phase 1] Discovery & Date Sanitization
    - Ingest player_data.csv.
    - Convert birth_date safely into datetime objects.
    """
    print("\n[PHASE 1] Discovery & Data Sanitization")
    print("-" * 65)
    
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.lower().str.replace(' ', '_')
    df = df.rename(columns={'player': 'name', 'pos': 'position', 'colleges': 'college'})
    df['birth_date'] = pd.to_datetime(df['birth_date'], errors='coerce')
    
    print(f"Dataset Shape: {df.shape}")
    print(f"Missing Birth Dates: {df['birth_date'].isnull().sum()}")
    
    return df


def phase_2_birthday_paradox_analysis(df):
    """
    [Phase 2] Birthday Paradox & Frequency Analysis
    - Format birth_date into Month Day strings and count frequencies.
    """
    print("\n[PHASE 2] Birthday Paradox & Frequency Analysis")
    print("-" * 65)
    
    bday_series = df['birth_date'].dt.strftime('%B %d')
    top_bdays = bday_series.value_counts()
    print("Top 5 Shared Birthdays in NBA History:\n", top_bdays.head(5))
    
    return df


def phase_3_player_lookup_system(df, player_name):
    """
    [Phase 3] Case-Insensitive Player Search
    - Filter df matching player_name regardless of uppercase/lowercase.
    """
    print(f"\n[PHASE 3] Player Lookup: Searching for '{player_name}'")
    print("-" * 65)
    
    match = df[df['name'].str.contains(player_name, case=False, na=False)]
    
    if match.empty:
        print(f"Player '{player_name}' not found.")
        return None
        
    player = match.iloc[0]
    print(f"Name: {player['name']}")
    print(f"Position: {player['position']}")
    print(f"College: {player['college']}")
    print(f"Birth Date: {player['birth_date'].strftime('%B %d, %Y')}")
    
    return player


def phase_4_shared_birthday_cohort(df, player):
    """
    [Phase 4] Shared Birthday Cohort Finder
    - Filter all other players sharing the target player's birth month and day.
    """
    print(f"\n[PHASE 4] Shared Birthday Cohort Finder")
    print("-" * 65)
    
    if player is None:
        return
        
    p_month = player['birth_date'].month
    p_day = player['birth_date'].day
    
    cohort = df[(df['birth_date'].dt.month == p_month) & (df['birth_date'].dt.day == p_day)]
    peers = cohort[cohort['name'] != player['name']]
    
    print(f"Total players sharing {player['birth_date'].strftime('%B %d')}: {len(cohort)}")
    print(peers[['name', 'birth_date', 'position']].to_string(index=False))


def phase_5_evaluation_and_summary(df):
    """
    [Phase 5] Summary Reporting
    - Count valid records and unique calendar days occupied.
    """
    print("\n[PHASE 5] Summary & Reporting")
    print("-" * 65)
    
    total_valid = df['birth_date'].dropna().count()
    bday_days = df['birth_date'].dt.strftime('%m-%d')
    unique_days = bday_days.nunique()
    
    print(f"Total Validated Players: {total_valid}")
    print(f"Unique Birthday Days: {unique_days} / 366")


# ==============================================================================
# THE FACADE ORCHESTRATOR
# ==============================================================================
def run_analytics_lifecycle():
    print("=" * 65)
    print("STARTING NBA BIRTHDAY & BIOGRAPHICAL LIFECYCLE (Out of 40 Marks)")
    print("=" * 65)
    
    target_player = "Michael Jordan"
    
    df = phase_1_discovery_and_sanitization()
    df = phase_2_birthday_paradox_analysis(df)
    player = phase_3_player_lookup_system(df, target_player)
    phase_4_shared_birthday_cohort(df, player)
    phase_5_evaluation_and_summary(df)


if __name__ == "__main__":
    run_analytics_lifecycle()
