"""
LAYER 1 (REVISED) — POSSESSION-LEVEL ASSIST CONTEXT ANALYSIS
=============================================================
The play-by-play data does NOT contain pass sequences (only ball-ending
events), so true "passes per possession" cannot be computed. Instead we
extract the strongest available possession-level signal: how shot success
and shot type relate to whether a shot was ASSISTED (i.e. created by a pass).

This directly tests Coach Samson's passing hypothesis at the shot level.

RUN THIS in Colab/locally on the full playbyplay_data_dump.csv.
Upload the 3 small output CSVs back to Claude.
"""

import pandas as pd
import numpy as np
import re

# ── UPDATE PATH (Colab: /content/drive/MyDrive/playbyplay_data_dump.csv) ──
filepath = "playbyplay_data_dump.csv"
# ─────────────────────────────────────────────────────────────────────────

print("Loading full play-by-play (chunked)...")
chunks = []
cols = ['gameId','type','description','quarter','remainingMinutes','playerId']
for chunk in pd.read_csv(filepath, usecols=lambda c: c in cols, chunksize=200000):
    # keep only made field goals (where assist info lives)
    chunks.append(chunk[chunk['type'].isin(['FieldGoalMade','FieldGoalMissed'])])
df = pd.concat(chunks, ignore_index=True)
print(f"Loaded {len(df):,} field-goal attempts\n")

# ── Feature extraction from description text ──
df['made']     = (df['type'] == 'FieldGoalMade').astype(int)
df['assisted'] = df['description'].str.contains('assisted by', na=False).astype(int)
df['is_three'] = df['description'].str.contains('three-pointer', na=False).astype(int)

def get_dist(d):
    m = re.search(r'(\d+)-foot', str(d))
    return int(m.group(1)) if m else np.nan
df['distance'] = df['description'].apply(get_dist)

made = df[df['made'] == 1].copy()

# ── OUTPUT 1: Assist rate by shot type ──
out1 = pd.DataFrame({
    'category': ['all_made','three_pointers','two_pointers'],
    'assist_rate': [
        made['assisted'].mean(),
        made[made['is_three']==1]['assisted'].mean(),
        made[made['is_three']==0]['assisted'].mean(),
    ],
    'n': [len(made), (made['is_three']==1).sum(), (made['is_three']==0).sum()],
})
out1.to_csv('pbp_assist_rate_by_type.csv', index=False)
print("=== Assist rate by shot type ===")
print(out1.round(3).to_string(index=False))

# ── OUTPUT 2: Assist rate by distance bucket ──
made['dist_bucket'] = pd.cut(made['distance'],
    bins=[0,3,10,16,23,50],
    labels=['0-3ft','3-10ft','10-16ft','16-23ft','23ft+'])
out2 = made.groupby('dist_bucket', observed=True)['assisted'].agg(['mean','count']).reset_index()
out2.columns = ['distance','assist_rate','n']
out2.to_csv('pbp_assist_rate_by_distance.csv', index=False)
print("\n=== Assist rate by distance ===")
print(out2.round(3).to_string(index=False))

# ── OUTPUT 3: Per-game assisted-FG share + win linkage prep ──
per_game = made.groupby('gameId').agg(
    made_fg      = ('made','sum'),
    assisted_fg  = ('assisted','sum'),
    three_made   = ('is_three','sum'),
).reset_index()
per_game['assisted_share'] = per_game['assisted_fg'] / per_game['made_fg']
per_game.to_csv('pbp_per_game_assist_share.csv', index=False)
print(f"\n=== Per-game assisted share: mean={per_game['assisted_share'].mean():.3f} ===")
print(f"Saved {len(per_game)} games")

print("\n\nUpload these 3 files back to Claude:")
print("  1. pbp_assist_rate_by_type.csv")
print("  2. pbp_assist_rate_by_distance.csv")
print("  3. pbp_per_game_assist_share.csv")
