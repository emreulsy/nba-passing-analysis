"""
LAYER 3 — HYPOTHESIS GENERATION (Subgroup Discovery)
=====================================================
Automatically discovers game conditions under which win rate deviates
significantly from the 50% baseline. This is the "discovery" engine —
it finds patterns no one specified in advance.

Target: win (binary)
Method: pysubgroup beam search, WRAcc quality measure
"""

import pandas as pd
import numpy as np
import pysubgroup as ps

# ---------------------------------------------------------------
# Build a rich team-game feature set
# ---------------------------------------------------------------
match = pd.read_csv('/home/claude/match.csv')

def reshape(m):
    rows = []
    for side, opp in [('home','away'), ('away','home')]:
        d = pd.DataFrame({
            'gameId': m['gameId'],
            'isHome': 1 if side=='home' else 0,
            'assists':  m[f'{side}Assists'],
            'rebounds': m[f'{side}Rebounds'],
            'steals':   m[f'{side}Steals'],
            'blocks':   m[f'{side}Blocks'],
            'score':    m[f'{side}Score'],
            'oppAssists':  m[f'{opp}Assists'],
            'oppRebounds': m[f'{opp}Rebounds'],
            'oppSteals':   m[f'{opp}Steals'],
            'oppBlocks':   m[f'{opp}Blocks'],
            'oppScore':    m[f'{opp}Score'],
        })
        rows.append(d)
    return pd.concat(rows, ignore_index=True)

df = reshape(match)
df['win'] = (df['score'] > df['oppScore']).astype(int)

# Differentials (relative performance — the real signal)
df['assist_diff'] = df['assists'] - df['oppAssists']
df['reb_diff']    = df['rebounds'] - df['oppRebounds']
df['steal_diff']  = df['steals']  - df['oppSteals']
df['block_diff']  = df['blocks']  - df['oppBlocks']

# Categorical bins for interpretable subgroups
def cat_diff(x):
    return pd.cut(x, bins=[-100,-5,-1,1,5,100],
                  labels=['much_fewer','fewer','equal','more','much_more'])

df['assists_vs_opp'] = cat_diff(df['assist_diff'])
df['rebounds_vs_opp'] = cat_diff(df['reb_diff'])
df['steals_vs_opp']   = cat_diff(df['steal_diff'])
df['blocks_vs_opp']   = cat_diff(df['block_diff'])

# Absolute level bins
df['assist_level'] = pd.cut(df['assists'], bins=[0,18,23,28,50],
                            labels=['low','medium','high','very_high'])
df['venue'] = df['isHome'].map({1:'home', 0:'away'})

target_mean = df['win'].mean()
print(f"Baseline win rate: {target_mean:.3f}\n")

# ---------------------------------------------------------------
# Subgroup Discovery
# ---------------------------------------------------------------
search_cols = ['assists_vs_opp','rebounds_vs_opp','steals_vs_opp',
               'blocks_vs_opp','assist_level','venue']
sd_df = df[search_cols + ['win']].copy()
for c in search_cols:
    sd_df[c] = sd_df[c].astype(str)

target = ps.BinaryTarget('win', 1)
searchspace = ps.create_selectors(sd_df, ignore=['win'])
task = ps.SubgroupDiscoveryTask(
    sd_df, target, searchspace,
    result_set_size=15,
    depth=3,
    qf=ps.WRAccQF())

print("Running subgroup discovery (beam search, depth=3)...\n")
result = ps.BeamSearch().execute(task)
res_df = result.to_dataframe()

print("="*80)
print("TOP SUBGROUPS — conditions where win rate deviates most from 50%")
print("="*80)

rows = []
for i, row in res_df.head(15).iterrows():
    sg = row['subgroup']
    cover = sg.covers(sd_df)
    size = cover.sum()
    sg_winrate = sd_df[cover]['win'].mean()
    lift = sg_winrate / target_mean
    rows.append({
        'rule': str(sg),
        'size': int(size),
        'coverage_%': round(100*size/len(sd_df),1),
        'win_rate': round(sg_winrate,3),
        'lift': round(lift,2),
    })

out = pd.DataFrame(rows)
pd.set_option('display.max_colwidth', 70)
pd.set_option('display.width', 200)
print(out.to_string(index=False))

# Also find LOSING subgroups (invert target)
print("\n" + "="*80)
print("TOP LOSING CONDITIONS — where win rate is LOWEST")
print("="*80)
target_lose = ps.BinaryTarget('win', 0)
task_lose = ps.SubgroupDiscoveryTask(
    sd_df, target_lose, searchspace,
    result_set_size=8, depth=3, qf=ps.WRAccQF())
result_lose = ps.BeamSearch().execute(task_lose)
res_lose = result_lose.to_dataframe()

rows = []
for i, row in res_lose.head(8).iterrows():
    sg = row['subgroup']
    cover = sg.covers(sd_df)
    size = cover.sum()
    sg_winrate = sd_df[cover]['win'].mean()
    rows.append({
        'rule': str(sg),
        'size': int(size),
        'win_rate': round(sg_winrate,3),
        'lift': round(sg_winrate/target_mean,2),
    })
print(pd.DataFrame(rows).to_string(index=False))

df.to_csv('/home/claude/team_game_features.csv', index=False)
print("\n[Saved feature set -> team_game_features.csv]")
