import pandas as pd
import numpy as np
from scipy import stats

p = pd.read_csv('player.csv', low_memory=False)
match = pd.read_csv('match.csv')

# Pozisyonları ana gruplara indir
pos_map = {'PG':'Guard','SG':'Guard','G':'Guard','GF':'Wing','SF':'Wing','F':'Wing',
           'PF':'Big','C':'Big','FC':'Big'}
p['pos_group'] = p['position'].map(pos_map)

print("="*65)
print("A — OYUNCU ROLÜ ANALİZİ")
print("="*65)
# Sadece oynayan oyuncular (minutes>0)
played = p[p['minutes']>0].copy()
print("\n[A1] Pozisyon grubuna göre maç başına ortalama asist:")
print(played.groupby('pos_group')['assists'].agg(['mean','sum','count']).round(2).to_string())

print("\n[A2] Toplam asistlerin pozisyona göre dağılımı:")
total_ast = played.groupby('pos_group')['assists'].sum()
for pg, v in total_ast.items():
    print(f"    {pg}: {v:,.0f} asist (%{v/total_ast.sum()*100:.1f})")

print("\n[A3] Detaylı pozisyon (orijinal) - ort asist:")
print(played.groupby('position')['assists'].mean().sort_values(ascending=False).round(2).to_string())

# ============================================================
print("\n" + "="*65)
print("B — ASIST DAĞILIM DENGESİ (paylaşımlı oyun)")
print("="*65)
# Her takım-maç için asistin oyuncular arası dağılımı (Gini / herfindahl)
def herfindahl(x):
    x = x[x>0]
    if x.sum()==0: return np.nan
    shares = x/x.sum()
    return (shares**2).sum()  # 1=tek kişi, düşük=dağıtılmış

team_game = played.groupby(['gameId','teamId']).agg(
    total_assists=('assists','sum'),
    n_assisters=('assists', lambda x: (x>0).sum()),
    hhi=('assists', herfindahl),
).reset_index()

# Galibiyet bilgisini ekle
home = match[['gameId','homeTeamId','homeScore','awayScore']].rename(columns={'homeTeamId':'teamId'})
home['win'] = (home['homeScore']>home['awayScore']).astype(int)
away = match[['gameId','awayTeamId','homeScore','awayScore']].rename(columns={'awayTeamId':'teamId'})
away['win'] = (away['awayScore']>away['homeScore']).astype(int)
outcomes = pd.concat([home[['gameId','teamId','win']], away[['gameId','teamId','win']]])

tg = team_game.merge(outcomes, on=['gameId','teamId'], how='inner')
print(f"\n[B1] Takım-maç sayısı: {len(tg):,}")
print(f"    Ortalama asist veren oyuncu sayısı: {tg['n_assisters'].mean():.1f}")
print(f"    Ortalama HHI (dağılım yoğunluğu): {tg['hhi'].mean():.3f}")

print("\n[B2] Asist dağılımı dengesi vs galibiyet:")
tg['balance'] = pd.cut(tg['hhi'], bins=[0,0.15,0.20,0.25,1.0],
                       labels=['çok dengeli','dengeli','orta','tek-merkezli'])
print(tg.groupby('balance', observed=True)['win'].agg(['mean','count']).round(3).to_string())

print("\n[B3] Asist veren oyuncu sayısı vs galibiyet:")
tg['n_bucket'] = pd.cut(tg['n_assisters'], bins=[0,4,6,8,20],
                        labels=['<=4','5-6','7-8','9+'])
print(tg.groupby('n_bucket', observed=True)['win'].agg(['mean','count']).round(3).to_string())

corr_n = tg['n_assisters'].corr(tg['win'])
print(f"\n    Asist veren sayısı - galibiyet korelasyonu: r={corr_n:.3f}")

# ============================================================
print("\n" + "="*65)
print("C — SAKATLIK ETKİSİ")
print("="*65)
# Her takım-maç için kaç oyuncu 'Out'
inj = p.copy()
inj['is_out'] = (inj['injuryStatus']=='Out').astype(int)
team_inj = inj.groupby(['gameId','teamId'])['is_out'].sum().reset_index()
team_inj = team_inj.merge(outcomes, on=['gameId','teamId'], how='inner')
print(f"\n[C1] Sakatlık (Out) durumu olan takım-maç: {(team_inj['is_out']>0).sum():,} / {len(team_inj):,}")
print("\n[C2] Sakat oyuncu sayısı vs galibiyet:")
team_inj['out_bucket'] = pd.cut(team_inj['is_out'], bins=[-1,0,1,2,20],
                                labels=['0 sakat','1 sakat','2 sakat','3+ sakat'])
print(team_inj.groupby('out_bucket', observed=True)['win'].agg(['mean','count']).round(3).to_string())
