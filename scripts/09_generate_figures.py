"""
NBA Passing & Winning Analysis — Figure Generation
Generates publication-quality figures for the report.
Author: [Your Name] | Sports Data Science, Leiden University (LIACS)
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

# ---- Professional styling ----
mpl.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.titleweight': 'bold',
    'axes.labelsize': 11,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': True,
    'grid.alpha': 0.25,
    'grid.linewidth': 0.6,
    'figure.dpi': 150,
    'savefig.dpi': 200,
    'savefig.bbox': 'tight',
})
NAVY='#1F3864'; BLUE='#2E5DA6'; TEAL='#1D9E75'; AMBER='#BA7517'; RED='#C0392B'; GRAY='#888780'
FIG='/home/claude/figures/'

# ---- Load & reshape ----
match = pd.read_csv('match.csv')
def reshape(m):
    rows=[]
    for s,o in [('home','away'),('away','home')]:
        rows.append(pd.DataFrame({
            'assists':m[f'{s}Assists'],'rebounds':m[f'{s}Rebounds'],'steals':m[f'{s}Steals'],
            'blocks':m[f'{s}Blocks'],'score':m[f'{s}Score'],'oppScore':m[f'{o}Score'],
            'oppAssists':m[f'{o}Assists'],'oppRebounds':m[f'{o}Rebounds'],
            'oppSteals':m[f'{o}Steals'],'oppBlocks':m[f'{o}Blocks'],'season':m['season']}))
    return pd.concat(rows,ignore_index=True)
df=reshape(match)
df['win']=(df['score']>df['oppScore']).astype(int)
df['assist_diff']=df['assists']-df['oppAssists']

# ===== FIG 1: Assists vs win rate =====
fig,ax=plt.subplots(figsize=(6,4))
df['ab']=pd.cut(df['assists'],bins=[0,15,20,25,30,50],labels=['<15','15-20','20-25','25-30','30+'])
wr=df.groupby('ab',observed=True)['win'].mean()*100
bars=ax.bar(range(len(wr)),wr.values,color=[ '#B5D4F4','#85B7EB','#378ADD','#185FA5','#0C447C'],width=0.65)
ax.set_xticks(range(len(wr)));ax.set_xticklabels(wr.index)
ax.set_xlabel('Assists per game');ax.set_ylabel('Win rate (%)')
ax.set_title('Win rate increases with assists',loc='left')
ax.axhline(50,color=GRAY,ls='--',lw=0.8)
for b,v in zip(bars,wr.values):ax.text(b.get_x()+b.get_width()/2,v+1.5,f'{v:.0f}%',ha='center',fontsize=10,fontweight='bold')
ax.set_ylim(0,100)
plt.savefig(FIG+'fig1_assists_winrate.png');plt.close()

# ===== FIG 2: Four-factor dominance =====
df['dom']=((df.assists>df.oppAssists).astype(int)+(df.rebounds>df.oppRebounds).astype(int)+
           (df.steals>df.oppSteals).astype(int)+(df.blocks>df.oppBlocks).astype(int))
fig,ax=plt.subplots(figsize=(6,4))
wr2=df.groupby('dom')['win'].mean()*100
cols=['#9FE1CB','#5DCAA5','#1D9E75','#0F6E56','#085041']
bars=ax.bar(wr2.index,wr2.values,color=cols,width=0.65)
ax.set_xlabel('Number of categories dominated (of 4)');ax.set_ylabel('Win rate (%)')
ax.set_title('Dominating statistical categories predicts winning',loc='left')
ax.axhline(50,color=GRAY,ls='--',lw=0.8)
for b,v in zip(bars,wr2.values):ax.text(b.get_x()+b.get_width()/2,v+1.5,f'{v:.0f}%',ha='center',fontsize=10,fontweight='bold')
ax.set_ylim(0,100)
plt.savefig(FIG+'fig2_four_factor.png');plt.close()

# ===== FIG 3: Feature importance =====
fig,ax=plt.subplots(figsize=(6,3.5))
feats=['Assist\ndifferential','Rebound\ndifferential','Steal\ndifferential','Block\ndifferential','Home\nadvantage']
imp=[44.3,31.0,13.1,9.6,2.0]
bars=ax.barh(feats[::-1],imp[::-1],color=['#D3D1C7','#9FE1CB','#5DCAA5','#1D9E75','#0F6E56'])
ax.set_xlabel('Feature importance (%)')
ax.set_title('Assist differential is the top predictor',loc='left')
for b,v in zip(bars,imp[::-1]):ax.text(v+0.8,b.get_y()+b.get_height()/2,f'{v:.0f}%',va='center',fontsize=10,fontweight='bold')
ax.set_xlim(0,50)
plt.savefig(FIG+'fig3_feature_importance.png');plt.close()

# ===== FIG 4: Assist rate by shot type (from pbp) =====
fig,ax=plt.subplots(figsize=(6,3.5))
cats=['Three-pointers','All made shots','Two-pointers']
rates=[82.5,59.0,50.3]
bars=ax.barh(cats[::-1],rates[::-1],color=['#9FE1CB','#5DCAA5','#0F6E56'])
ax.set_xlabel('Share of made shots that were assisted (%)')
ax.set_title('Three-pointers are overwhelmingly assisted',loc='left')
for b,v in zip(bars,rates[::-1]):ax.text(v+1,b.get_y()+b.get_height()/2,f'{v:.0f}%',va='center',fontsize=10,fontweight='bold')
ax.set_xlim(0,100)
plt.savefig(FIG+'fig4_assist_shot_type.png');plt.close()

# ===== FIG 5: Circularity control =====
fig,ax=plt.subplots(figsize=(6,3))
labels=['Raw correlation','Controlling for points']
vals=[0.314,0.082]
bars=ax.barh(labels[::-1],vals[::-1],color=['#5DCAA5','#0F6E56'])
ax.set_xlabel('Correlation (assists vs win)')
ax.set_title('A real signal remains after controlling for scoring',loc='left')
for b,v in zip(bars,vals[::-1]):ax.text(v+0.008,b.get_y()+b.get_height()/2,f'r = {v:.3f}',va='center',fontsize=10,fontweight='bold')
ax.set_xlim(0,0.4)
plt.savefig(FIG+'fig5_circularity.png');plt.close()

# ===== FIG 6: Season trend =====
fig,ax=plt.subplots(figsize=(6,3.5))
seasons=list(range(2009,2021))
assists=[21.0,20.6,21.4,21.0,22.1,22.0,22.0,22.3,22.6,23.2,24.6,24.3]
ax.plot(seasons,assists,marker='o',color=BLUE,lw=2,markersize=5)
ax.set_xlabel('Season');ax.set_ylabel('Avg assists per team-game')
ax.set_title('Assists rose with the three-point revolution',loc='left')
ax.set_ylim(19,26)
plt.savefig(FIG+'fig6_season_trend.png');plt.close()

# ===== FIG 7: Quarter importance =====
fig,ax=plt.subplots(figsize=(6,3.5))
q=['Quarter 1','Quarter 2','Quarter 3','Quarter 4']
qc=[0.456,0.434,0.469,0.355]
bars=ax.bar(q,qc,color=['#85B7EB','#85B7EB','#185FA5','#B5D4F4'],width=0.6)
ax.set_ylabel('Correlation with final margin')
ax.set_title('The third quarter matters most',loc='left')
for b,v in zip(bars,qc):ax.text(b.get_x()+b.get_width()/2,v+0.008,f'{v:.2f}',ha='center',fontsize=10,fontweight='bold')
ax.set_ylim(0,0.5)
plt.savefig(FIG+'fig7_quarter.png');plt.close()

# ===== FIG 8: Number of assisters vs win =====
fig,ax=plt.subplots(figsize=(6,3.5))
nb=['≤4','5-6','7-8','9+']
nw=[41.3,44.6,50.7,55.7]
bars=ax.bar(nb,nw,color=['#F0997B','#FAC775','#9FE1CB','#1D9E75'],width=0.6)
ax.set_xlabel('Number of players recording an assist')
ax.set_ylabel('Win rate (%)')
ax.set_title('Sharing the ball across more players wins games',loc='left')
ax.axhline(50,color=GRAY,ls='--',lw=0.8)
for b,v in zip(bars,nw):ax.text(b.get_x()+b.get_width()/2,v+1.5,f'{v:.0f}%',ha='center',fontsize=10,fontweight='bold')
ax.set_ylim(0,70)
plt.savefig(FIG+'fig8_assisters.png');plt.close()

print("All 8 figures generated successfully")
import os
for f in sorted(os.listdir(FIG)):print("  -",f)
