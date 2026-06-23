"""
NBA Winning Conditions Analysis
================================
Q0: Does more passing (assists) increase win probability?
Q3: Does dominating the 4 statistical categories predict winning?

Approach: Treat every team-game as one observation (home + away = 2x rows).
This doubles the sample and removes home/away bias from the core analysis.
"""

import pandas as pd
import numpy as np

# ---------------------------------------------------------------
# 1. LOAD & RESHAPE — one row per team per game
# ---------------------------------------------------------------
match = pd.read_csv('match_data_dump.csv')
print(f"Loaded {len(match):,} games\n")

home = pd.DataFrame({
    'gameId':   match['gameId'],
    'season':   match['season'],
    'team':     match['homeTeam'],
    'opponent': match['awayTeam'],
    'isHome':   1,
    'assists':  match['homeAssists'],
    'rebounds': match['homeRebounds'],
    'steals':   match['homeSteals'],
    'blocks':   match['homeBlocks'],
    'score':    match['homeScore'],
    'oppScore': match['awayScore'],
    'oppAssists':  match['awayAssists'],
    'oppRebounds': match['awayRebounds'],
    'oppSteals':   match['awaySteals'],
    'oppBlocks':   match['awayBlocks'],
})

away = pd.DataFrame({
    'gameId':   match['gameId'],
    'season':   match['season'],
    'team':     match['awayTeam'],
    'opponent': match['homeTeam'],
    'isHome':   0,
    'assists':  match['awayAssists'],
    'rebounds': match['awayRebounds'],
    'steals':   match['awaySteals'],
    'blocks':   match['awayBlocks'],
    'score':    match['awayScore'],
    'oppScore': match['homeScore'],
    'oppAssists':  match['homeAssists'],
    'oppRebounds': match['homeRebounds'],
    'oppSteals':   match['homeSteals'],
    'oppBlocks':   match['homeBlocks'],
})

df = pd.concat([home, away], ignore_index=True)
df['win'] = (df['score'] > df['oppScore']).astype(int)
print(f"Reshaped to {len(df):,} team-game observations")
print(f"Overall win rate: {df['win'].mean():.3f} (should be ~0.5)\n")

# ---------------------------------------------------------------
# 2. Q0 — ASSISTS vs WIN RATE
# ---------------------------------------------------------------
print("="*60)
print("Q0: ASSISTS (passing proxy) vs WIN RATE")
print("="*60)

df['assist_bucket'] = pd.cut(df['assists'],
    bins=[0,15,20,25,30,50],
    labels=['<15','15-20','20-25','25-30','30+'])
q0 = df.groupby('assist_bucket', observed=True)['win'].agg(['mean','count'])
q0.columns = ['win_rate','games']
print(q0.round(3).to_string())

corr = df['assists'].corr(df['win'])
print(f"\nCorrelation (assists vs win): {corr:.3f}")

# Assist DIFFERENTIAL (relative to opponent) is even stronger
df['assist_diff'] = df['assists'] - df['oppAssists']
df['assist_diff_bucket'] = pd.cut(df['assist_diff'],
    bins=[-50,-10,-5,0,5,10,50],
    labels=['<-10','-10..-5','-5..0','0..5','5..10','10+'])
print("\n--- Assist DIFFERENTIAL vs win rate ---")
q0d = df.groupby('assist_diff_bucket', observed=True)['win'].agg(['mean','count'])
q0d.columns = ['win_rate','games']
print(q0d.round(3).to_string())
print(f"Correlation (assist_diff vs win): {df['assist_diff'].corr(df['win']):.3f}")

# ---------------------------------------------------------------
# 3. Q3 — FOUR-FACTOR DOMINANCE
# ---------------------------------------------------------------
print("\n" + "="*60)
print("Q3: FOUR-FACTOR DOMINANCE vs WIN RATE")
print("="*60)

df['dom_assists']  = (df['assists']  > df['oppAssists']).astype(int)
df['dom_rebounds'] = (df['rebounds'] > df['oppRebounds']).astype(int)
df['dom_steals']   = (df['steals']   > df['oppSteals']).astype(int)
df['dom_blocks']   = (df['blocks']   > df['oppBlocks']).astype(int)
df['cats_dominated'] = df[['dom_assists','dom_rebounds','dom_steals','dom_blocks']].sum(axis=1)

q3 = df.groupby('cats_dominated')['win'].agg(['mean','count'])
q3.columns = ['win_rate','games']
print(q3.round(3).to_string())

# Which single category matters most?
print("\n--- Win rate when dominating EACH category individually ---")
for cat, col in [('Assists','dom_assists'),('Rebounds','dom_rebounds'),
                 ('Steals','dom_steals'),('Blocks','dom_blocks')]:
    wr = df[df[col]==1]['win'].mean()
    print(f"  {cat:10s}: {wr:.3f}")

# ---------------------------------------------------------------
# 4. CORRELATION SUMMARY
# ---------------------------------------------------------------
print("\n" + "="*60)
print("CORRELATIONS WITH WINNING")
print("="*60)
cols = ['assists','rebounds','steals','blocks',
        'assist_diff','isHome']
for c in cols:
    print(f"  {c:14s}: {df[c].corr(df['win']):+.3f}")

# Save the reshaped dataset for later modelling
df.to_csv('team_game_analysis.csv', index=False)
print("\nSaved reshaped dataset -> team_game_analysis.csv")
