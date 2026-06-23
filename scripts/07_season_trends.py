import pandas as pd
import numpy as np

match = pd.read_csv('match.csv')

def reshape(m):
    rows = []
    for side, opp in [('home','away'), ('away','home')]:
        rows.append(pd.DataFrame({
            'season': m['season'],
            'status': m['status'],
            'assists':  m[f'{side}Assists'],
            'score':    m[f'{side}Score'],
            'oppScore': m[f'{opp}Score'],
            'oppAssists': m[f'{opp}Assists'],
        }))
    return pd.concat(rows, ignore_index=True)

df = reshape(match)
df['win'] = (df['score'] > df['oppScore']).astype(int)
df['assist_diff'] = df['assists'] - df['oppAssists']

print("="*65)
print("ROBUSTLUK & SEZON TRENDİ ANALİZİ")
print("="*65)

print("\n[1] Sezonlara göre ortalama asist (3 sayı devrimi etkisi):")
season_stats = df.groupby('season').agg(
    avg_assists=('assists','mean'),
    avg_score=('score','mean'),
).round(1)
for s, row in season_stats.iterrows():
    print(f"    {s}: ort asist={row['avg_assists']:.1f}  ort sayı={row['avg_score']:.1f}")

print("\n[2] Asist-galibiyet ilişkisi her sezon tutarlı mı? (robustluk)")
for s in sorted(df['season'].unique()):
    sub = df[df['season']==s]
    r = sub['assist_diff'].corr(sub['win'])
    print(f"    {s}: r={r:.3f}")

print("\n[3] Overtime maçlarında ilişki değişiyor mu?")
reg = df[df['status']=='Final']
ot = df[df['status']=='F/OT']
print(f"    Normal maçlar (n={len(reg):,}): asist_diff-win r={reg['assist_diff'].corr(reg['win']):.3f}")
print(f"    Uzatmalı maçlar (n={len(ot):,}): asist_diff-win r={ot['assist_diff'].corr(ot['win']):.3f}")

print("\n[4] İlk dönem (2009-2014) vs son dönem (2015-2020):")
early = df[df['season']<=2014]
late = df[df['season']>=2015]
print(f"    2009-2014: ort asist={early['assists'].mean():.1f}, asist-win r={early['assist_diff'].corr(early['win']):.3f}")
print(f"    2015-2020: ort asist={late['assists'].mean():.1f}, asist-win r={late['assist_diff'].corr(late['win']):.3f}")
print(f"    -> Asist sayısı arttı mı? {'EVET' if late['assists'].mean()>early['assists'].mean() else 'HAYIR'} (+{late['assists'].mean()-early['assists'].mean():.1f})")
