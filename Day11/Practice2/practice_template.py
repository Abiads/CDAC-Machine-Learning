"""
================================================================================
PRACTICE TEMPLATE: Day 11 - Practice 2 (NBA Player Birthday Paradox & Lookup)
================================================================================
INSTRUCTIONS FOR THE TEST:
In the exam, you are given this skeleton. Follow the TODOs step-by-step.
Run this script using: python practice_template.py

DATASET: player_data.csv
COLUMNS: ['name', 'year_start', 'year_end', 'position', 'height', 'weight', 'birth_date', 'college']
KEY TASKS:
1. pd.to_datetime with errors='coerce'
2. dt.strftime('%B %d').value_counts()
3. Case-insensitive lookup (.str.lower() == target.lower())
4. Cohort filtering (dt.month == target_month & dt.day == target_day)
================================================================================
"""

from pathlib import Path
# TODO: Step 0 - Uncomment and import required libraries
# import pandas as pd

# PATH TRAP AVOIDANCE
csv_path = Path(__file__).resolve().parent / "player_data.csv"


def phase_1_discovery_and_sanitization():
    """
    [Phase 1] Discovery & Data Sanitization (5 Marks)
    -------------------------------------------------------------
    WHERE & WHAT TO CHANGE:
    1. Read CSV: df = pd.read_csv(csv_path)
    2. Convert birth_date to datetime:
       # TRAP: Always pass errors='coerce' so corrupted date strings become NaT instead of crashing!
       df['birth_date'] = pd.to_datetime(df['birth_date'], errors='coerce')
    3. Check shape and nulls:
       print(f"Shape: {df.shape}")
       print("Missing Birth Dates:", df['birth_date'].isnull().sum())
    4. CRITICAL: Always return df!
    """
    print("\n[PHASE 1] Discovery & Data Sanitization")
    print("-" * 65)
    
    # --- WRITE YOUR CODE HERE ---
    # df = pd.read_csv(csv_path)
    # df['birth_date'] = pd.to_datetime(df['birth_date'], errors='coerce')
    # print(f"Shape: {df.shape}")
    # print(f"Missing Birth Dates: {df['birth_date'].isnull().sum()}")
    # return df
    pass


def phase_2_birthday_paradox_analysis(df):
    """
    [Phase 2] Birthday Paradox & Frequency Analysis (10 Marks)
    -------------------------------------------------------------
    WHERE & WHAT TO CHANGE:
    1. Extract calendar birthday string:
       bday_str = df['birth_date'].dt.strftime('%B %d')
    2. Count frequency per calendar day:
       counts = bday_str.value_counts()
       print("Top 5 Shared Birthdays:\n", counts.head(5))
    3. CRITICAL: Always return df!
    """
    print("\n[PHASE 2] Birthday Paradox & Frequency Analysis")
    print("-" * 65)
    
    # --- WRITE YOUR CODE HERE ---
    # bday_series = df['birth_date'].dt.strftime('%B %d')
    # top_bdays = bday_series.value_counts()
    # print("Top 5 Shared Birthdays in NBA History:\n", top_bdays.head(5))
    # return df
    pass


def phase_3_player_lookup_system(df, player_name):
    """
    [Phase 3] Case-Insensitive Player Lookup (10 Marks)
    -------------------------------------------------------------
    WHERE & WHAT TO CHANGE:
    1. Filter row matching player_name (case-insensitive):
       # TRAP: Use .str.lower() on column and .lower() on input string!
       match = df[df['name'].str.lower() == player_name.lower()]
    2. Check if player found:
       if match.empty:
           print(f"Player '{player_name}' not found.")
           return None
       player = match.iloc[0]
       print(f"Player: {player['name']}, Position: {player['position']}, Birth Date: {player['birth_date'].strftime('%B %d, %Y')}")
       return player
    """
    print(f"\n[PHASE 3] Player Lookup: Searching for '{player_name}'")
    print("-" * 65)
    
    # --- WRITE YOUR CODE HERE ---
    # match = df[df['name'].str.lower() == player_name.lower()]
    # if match.empty:
    #     print(f"Player '{player_name}' not found.")
    #     return None
    # player = match.iloc[0]
    # print(f"Name: {player['name']}")
    # print(f"Position: {player['position']}")
    # print(f"College: {player['college']}")
    # print(f"Birth Date: {player['birth_date'].strftime('%B %d, %Y')}")
    # return player
    pass


def phase_4_shared_birthday_cohort(df, player):
    """
    [Phase 4] Shared Birthday Cohort Finder (10 Marks)
    -------------------------------------------------------------
    WHERE & WHAT TO CHANGE:
    1. Extract player's month and day:
       p_month = player['birth_date'].month
       p_day = player['birth_date'].day
    2. Query dataset for all players sharing this month and day:
       cohort = df[(df['birth_date'].dt.month == p_month) & (df['birth_date'].dt.day == p_day)]
       peers = cohort[cohort['name'] != player['name']]
       print(f"Total players sharing {player['birth_date'].strftime('%B %d')}: {len(cohort)}")
       print("Peer Names:\n", peers[['name', 'birth_date', 'position']].to_string(index=False))
    """
    print(f"\n[PHASE 4] Shared Birthday Cohort Finder")
    print("-" * 65)
    
    # --- WRITE YOUR CODE HERE ---
    # if player is None:
    #     return
    # p_month, p_day = player['birth_date'].month, player['birth_date'].day
    # cohort = df[(df['birth_date'].dt.month == p_month) & (df['birth_date'].dt.day == p_day)]
    # peers = cohort[cohort['name'] != player['name']]
    # print(f"Total players born on {player['birth_date'].strftime('%B %d')}: {len(cohort)}")
    # print(peers[['name', 'birth_date', 'position']].to_string(index=False))
    pass


def phase_5_evaluation_and_summary(df):
    """
    [Phase 5] Summary & Reporting (5 Marks)
    -------------------------------------------------------------
    Print concluding insights on Birthday Paradox proof in the NBA.
    """
    print("\n[PHASE 5] Summary & Reporting")
    print("-" * 65)
    # --- WRITE YOUR CODE HERE ---
    # total_players = len(df.dropna(subset=['birth_date']))
    # print(f"Total validated player records: {total_players}")
    # print("Birthday Paradox Verified: Shared dates are prevalent across 4,900+ players.")
    pass


# ==============================================================================
# THE FACADE ORCHESTRATOR
# TRAP: Uncomment lines chronologically as you complete each phase!
# ==============================================================================
def run_analytics_lifecycle():
    print("=" * 65)
    print("STARTING NBA BIRTHDAY & BIOGRAPHICAL LIFECYCLE (Out of 40 Marks)")
    print("=" * 65)
    
    target_player = "Kareem Abdul-Jabbar"
    
    df = phase_1_discovery_and_sanitization()
    # df = phase_2_birthday_paradox_analysis(df)
    # player = phase_3_player_lookup_system(df, target_player)
    # phase_4_shared_birthday_cohort(df, player)
    # phase_5_evaluation_and_summary(df)


if __name__ == "__main__":
    run_analytics_lifecycle()
