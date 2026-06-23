import pandas as pd
import numpy as np
from scipy import stats

match = pd.read_csv('match.csv')
m = match.copy()
m['homeWin'] = (m['pointsDiff'] > 0).astype(int)

# Çeyrek bazında ev sahibi farkı
for q in range(4):
    m[f'q{q}_diff'] = m[f'quarter{q}home'] - m[f'quarter{q}away']
    m[f'q{q}_homeWon'] = (m[f'q{q}_diff'] > 0).astype(int)

print("="*65)
print("ÇEYREK BAZLI MOMENTUM ANALİZİ")
print("="*65)

print("\n[1] Her çeyreği kazanmanın maçı kazanmayla ilişkisi:")
for q in range(4):
    # o çeyreği kazandıysa maçı kazanma oranı
    won_q = m[m[f'q{q}_homeWon']==1]['homeWin'].mean()
    corr = m[f'q{q}_diff'].corr(m['homeWin'])
    print(f"    Çeyrek {q+1}: çeyreği kazanınca maç kazanma=%{won_q*100:.0f}  |  korelasyon r={corr:.3f}")

print("\n[2] Hangi çeyreğin farkı maç sonucunu en çok belirliyor?")
diffs = [f'q{q}_diff' for q in range(4)]
corrs = {f'Çeyrek {q+1}': m[f'q{q}_diff'].corr(m['pointsDiff']) for q in range(4)}
for k,v in sorted(corrs.items(), key=lambda x:-x[1]):
    print(f"    {k}: r={v:.3f} (toplam sayı farkıyla)")

print("\n[3] İlk yarı vs ikinci yarı önemi:")
m['firstHalf_diff'] = m['q0_diff'] + m['q1_diff']
m['secondHalf_diff'] = m['q2_diff'] + m['q3_diff']
print(f"    İlk yarı farkı -> galibiyet: r={m['firstHalf_diff'].corr(m['homeWin']):.3f}")
print(f"    İkinci yarı farkı -> galibiyet: r={m['secondHalf_diff'].corr(m['homeWin']):.3f}")

print("\n[4] GERİ DÖNÜŞ analizi — yarıda geride olup kazanma:")
m['halftime_lead'] = m['firstHalf_diff']
behind_half = m[m['halftime_lead'] < 0]
comeback_rate = behind_half['homeWin'].mean()
print(f"    Devre arası geride olan takımların kazanma oranı: %{comeback_rate*100:.1f}")
# Ne kadar geride?
for margin in [(-5,0),(-10,-5),(-100,-10)]:
    sub = m[(m['halftime_lead']>=margin[0]) & (m['halftime_lead']<margin[1])]
    print(f"    Devre arası {abs(margin[1])}-{abs(margin[0])} geride: kazanma=%{sub['homeWin'].mean()*100:.1f} (n={len(sub)})")

print("\n[5] 3. çeyrek 'şampiyonluk çeyreği' mi? (clutch quarter)")
# 4. çeyreğe eşit/yakın girip kazananlar
m['after3q'] = m['q0_diff']+m['q1_diff']+m['q2_diff']
close_after3 = m[abs(m['after3q'])<=3]
print(f"    3 çeyrek sonunda fark <=3 olan maçlar: {len(close_after3)}")
print(f"    Bu maçlarda 4. çeyreği kazanan maçı kazanma oranı: %{(close_after3['q3_homeWon']==close_after3['homeWin']).mean()*100:.1f}")
