"""
LAYER 2 — STATISTICAL HYPOTHESIS TESTING
=========================================
Rigorously tests Coach Samson's two hypotheses with proper statistics:
  H1: More passing (assists) increases win probability
  H3: Dominating the four factors predicts winning

Tests used:
  - Point-biserial correlation (continuous vs binary win)
  - Independent t-test (assists: winners vs losers)
  - Chi-square test (dominance categories vs win/loss)
  - Logistic regression (effect size + significance)
  - Cohen's d (practical effect size)
"""

import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm

match = pd.read_csv('/home/claude/match.csv')

# Reshape: one row per team-game
def reshape(m):
    rows = []
    for side, opp in [('home','away'), ('away','home')]:
        rows.append(pd.DataFrame({
            'assists':  m[f'{side}Assists'],
            'rebounds': m[f'{side}Rebounds'],
            'steals':   m[f'{side}Steals'],
            'blocks':   m[f'{side}Blocks'],
            'oppAssists':  m[f'{opp}Assists'],
            'oppRebounds': m[f'{opp}Rebounds'],
            'oppSteals':   m[f'{opp}Steals'],
            'oppBlocks':   m[f'{opp}Blocks'],
            'score':    m[f'{side}Score'],
            'oppScore': m[f'{opp}Score'],
        }))
    return pd.concat(rows, ignore_index=True)

df = reshape(match)
df['win'] = (df['score'] > df['oppScore']).astype(int)
df['assist_diff'] = df['assists'] - df['oppAssists']
N = len(df)

print("="*65)
print(f"SAMPLE: {N:,} team-game observations | Win rate: {df['win'].mean():.3f}")
print("="*65)

# ================================================================
# H1: ASSISTS vs WINNING
# ================================================================
print("\n" + "#"*65)
print("# HYPOTHESIS 1: More passing (assists) -> higher win probability")
print("#"*65)

# 1a. Point-biserial correlation
r_pb, p_pb = stats.pointbiserialr(df['win'], df['assists'])
print(f"\n[1a] Point-biserial correlation (assists vs win)")
print(f"     r = {r_pb:.4f},  p = {p_pb:.2e}")
print(f"     -> {'SIGNIFICANT' if p_pb < 0.05 else 'not significant'} at alpha=0.05")

# 1b. Independent t-test: assists of winners vs losers
win_assists  = df[df['win']==1]['assists']
lose_assists = df[df['win']==0]['assists']
t_stat, p_t = stats.ttest_ind(win_assists, lose_assists, equal_var=False)
# Cohen's d
pooled_sd = np.sqrt((win_assists.var() + lose_assists.var())/2)
cohens_d = (win_assists.mean() - lose_assists.mean()) / pooled_sd
print(f"\n[1b] Welch's t-test: assists of winners vs losers")
print(f"     Winners mean: {win_assists.mean():.2f}  |  Losers mean: {lose_assists.mean():.2f}")
print(f"     t = {t_stat:.2f},  p = {p_t:.2e}")
print(f"     Cohen's d = {cohens_d:.3f} ({'small' if abs(cohens_d)<0.5 else 'medium' if abs(cohens_d)<0.8 else 'large'} effect)")

# 1c. Assist differential is the stronger signal
r_pb2, p_pb2 = stats.pointbiserialr(df['win'], df['assist_diff'])
print(f"\n[1c] Point-biserial correlation (assist DIFFERENTIAL vs win)")
print(f"     r = {r_pb2:.4f},  p = {p_pb2:.2e}")
print(f"     -> assist_diff is a STRONGER predictor than raw assists ({r_pb2:.3f} vs {r_pb:.3f})")

# ================================================================
# H3: FOUR-FACTOR DOMINANCE
# ================================================================
print("\n" + "#"*65)
print("# HYPOTHESIS 3: Dominating the four factors predicts winning")
print("#"*65)

df['dom_assists']  = (df['assists']  > df['oppAssists']).astype(int)
df['dom_rebounds'] = (df['rebounds'] > df['oppRebounds']).astype(int)
df['dom_steals']   = (df['steals']   > df['oppSteals']).astype(int)
df['dom_blocks']   = (df['blocks']   > df['oppBlocks']).astype(int)
df['cats_dom'] = df[['dom_assists','dom_rebounds','dom_steals','dom_blocks']].sum(axis=1)

# 3a. Chi-square test: cats_dominated vs win
contingency = pd.crosstab(df['cats_dom'], df['win'])
chi2, p_chi, dof, expected = stats.chi2_contingency(contingency)
n = contingency.values.sum()
cramers_v = np.sqrt(chi2 / (n * (min(contingency.shape)-1)))
print(f"\n[3a] Chi-square test: categories dominated vs win/loss")
print(f"     chi2 = {chi2:.1f},  dof = {dof},  p = {p_chi:.2e}")
print(f"     Cramer's V = {cramers_v:.3f} ({'strong' if cramers_v>0.5 else 'moderate' if cramers_v>0.3 else 'weak'} association)")

# 3b. Each factor's individual chi-square + odds ratio
print(f"\n[3b] Individual factor tests (chi-square + odds ratio)")
for name, col in [('Assists','dom_assists'),('Rebounds','dom_rebounds'),
                  ('Steals','dom_steals'),('Blocks','dom_blocks')]:
    ct = pd.crosstab(df[col], df['win'])
    chi2_i, p_i, _, _ = stats.chi2_contingency(ct)
    # odds ratio
    a, b = ct.loc[1,1], ct.loc[1,0]   # dominate & win, dominate & lose
    c, d = ct.loc[0,1], ct.loc[0,0]   # not dom & win, not dom & lose
    odds_ratio = (a*d)/(b*c)
    print(f"     {name:9s}: chi2={chi2_i:7.1f}, p={p_i:.1e}, OR={odds_ratio:5.2f}")

# ================================================================
# MULTIVARIATE: LOGISTIC REGRESSION
# ================================================================
print("\n" + "#"*65)
print("# MULTIVARIATE LOGISTIC REGRESSION (all four differentials)")
print("#"*65)

df['reb_diff']   = df['rebounds'] - df['oppRebounds']
df['steal_diff'] = df['steals']   - df['oppSteals']
df['block_diff'] = df['blocks']   - df['oppBlocks']

X = df[['assist_diff','reb_diff','steal_diff','block_diff']]
X = sm.add_constant(X)
y = df['win']
model = sm.Logit(y, X).fit(disp=0)
print(f"\nPseudo R-squared: {model.prsquared:.4f}")
print(f"\n{'Variable':<14}{'Coef':>10}{'OddsRatio':>12}{'p-value':>12}")
for var in X.columns:
    coef = model.params[var]
    or_ = np.exp(coef)
    p = model.pvalues[var]
    print(f"{var:<14}{coef:>10.4f}{or_:>12.3f}{p:>12.2e}")

print("\nInterpretation: each +1 assist differential multiplies win odds by",
      f"{np.exp(model.params['assist_diff']):.2f}x")

# Save augmented dataset
df.to_csv('/home/claude/team_game_full.csv', index=False)
print("\n[Saved augmented dataset -> team_game_full.csv]")
