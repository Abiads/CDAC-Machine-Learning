"""
================================================================================
EXAM PRACTICE (FILL IN THE BLANKS): Day 11 - Practice 2
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

Fill in all the `___` blanks below to complete the implementation.
================================================================================
"""

from pathlib import Path
import pandas as pd

# Placeholder for your answers
___ = None

csv_path = Path(__file__).resolve().parent / "player_data.csv"


def phase_1_discovery_and_sanitization():
    """
    [Phase 1] Discovery & Date Sanitization
    - Ingest player_data.csv.
    - Convert birth_date safely into datetime objects.
    """
    print("\n[PHASE 1] Discovery & Data Sanitization")
    print("-" * 65)
    
    # FILL IN THE BLANKS:
    df = pd.read_csv(___)
    
    # 1. Which parameter safely converts bad date strings into NaT without crashing?
    df['birth_date'] = pd.to_datetime(df['birth_date'], errors=___)
    
    print(f"Dataset Shape: {df.shape}")
    print(f"Missing Birth Dates: {df['birth_date'].isnull().sum()}")
    
    return ___                          # 2. Return the sanitized DataFrame


def phase_2_birthday_paradox_analysis(df):
    """
    [Phase 2] Birthday Paradox & Frequency Analysis
    - Format birth_date into Month Day strings and count frequencies.
    """
    print("\n[PHASE 2] Birthday Paradox & Frequency Analysis")
    print("-" * 65)
    
    # FILL IN THE BLANKS:
    # 3. Format into Month Day string (e.g., 'August 23', '%B %d')
    bday_series = df['birth_date'].dt.strftime(___)
    
    # 4. Count frequency per calendar day
    top_bdays = bday_series.___()
    print("Top 5 Shared Birthdays in NBA History:\n", top_bdays.head(5))
    
    return ___                          # 5. Return the DataFrame


def phase_3_player_lookup_system(df, player_name):
    """
    [Phase 3] Case-Insensitive Player Search
    - Filter df matching player_name regardless of uppercase/lowercase.
    """
    print(f"\n[PHASE 3] Player Lookup: Searching for '{player_name}'")
    print("-" * 65)
    
    # FILL IN THE BLANKS:
    # 6. Apply lowercase string methods to both the dataframe column and player_name
    match = df[df['name'].str.___() == player_name.___()]
    
    if match.empty:
        print(f"Player '{player_name}' not found.")
        return None
        
    player = match.iloc[___]            # 7. Extract the first matching row (index 0)
    print(f"Name: {player['name']}")
    print(f"Position: {player['position']}")
    print(f"College: {player['college']}")
    print(f"Birth Date: {player['birth_date'].strftime('%B %d, %Y')}")
    
    return ___                          # 8. Return the player record


def phase_4_shared_birthday_cohort(df, player):
    """
    [Phase 4] Shared Birthday Cohort Finder
    - Filter all other players sharing the target player's birth month and day.
    """
    print(f"\n[PHASE 4] Shared Birthday Cohort Finder")
    print("-" * 65)
    
    if player is None:
        return
        
    # FILL IN THE BLANKS:
    p_month = player['birth_date'].___   # 9. Extract target player's month integer
    p_day = player['birth_date'].___     # 10. Extract target player's day integer
    
    # 11. Complete the condition matching birth month and birth day
    cohort = df[(df['birth_date'].dt.month == ___) & (df['birth_date'].dt.day == ___)]
    
    # 12. Exclude the target player from their peer list
    peers = cohort[cohort['name'] != player[___]]
    
    print(f"Total players sharing {player['birth_date'].strftime('%B %d')}: {len(cohort)}")
    print(peers[['name', 'birth_date', 'position']].to_string(index=False))


def phase_5_evaluation_and_summary(df):
    """
    [Phase 5] Summary Reporting
    - Count valid records and unique calendar days occupied.
    """
    print("\n[PHASE 5] Summary & Reporting")
    print("-" * 65)
    
    # FILL IN THE BLANKS:
    # 13. Count non-null birthdays
    total_valid = df['birth_date'].dropna().___()
    
    # 14. Count distinct birthday calendar days
    bday_days = df['birth_date'].dt.strftime('%m-%d')
    unique_days = bday_days.___()
    
    print(f"Total Validated Players: {total_valid}")
    print(f"Unique Birthday Days: {unique_days} / 366")


# ==============================================================================
# THE FACADE ORCHESTRATOR
# ==============================================================================
def run_analytics_lifecycle():
    print("=" * 65)
    print("STARTING NBA BIRTHDAY & BIOGRAPHICAL LIFECYCLE (Out of 40 Marks)")
    print("=" * 65)
    
    # Target player requested in this new scenario:
    target_player = "Michael Jordan"
    
    # df = phase_1_discovery_and_sanitization()
    # df = phase_2_birthday_paradox_analysis(df)
    # player = phase_3_player_lookup_system(df, target_player)
    # phase_4_shared_birthday_cohort(df, player)
    # phase_5_evaluation_and_summary(df)


if __name__ == "__main__":
    run_analytics_lifecycle()
