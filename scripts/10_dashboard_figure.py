import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.patches import FancyBboxPatch, Rectangle
mpl.rcParams.update({'font.family':'DejaVu Sans','savefig.dpi':200,'savefig.bbox':'tight'})

fig=plt.figure(figsize=(9,5.6))
fig.patch.set_facecolor('white')
ax=fig.add_axes([0,0,1,1]); ax.set_xlim(0,100); ax.set_ylim(0,100); ax.axis('off')

NAVY='#1F3864'; BLUE='#2E5DA6'; LGRAY='#F1EFE8'; GRAY='#888780'; DARK='#2C2C2A'

ax.text(3,96,'Coach Dashboard — Winning Conditions for The Hague Royals',fontsize=14,fontweight='bold',color=NAVY,va='top')
ax.text(3,91.5,'Based on 11 seasons of NBA data (2009-2020) and 12,777 games',fontsize=9.5,color=GRAY,va='top')
ax.plot([3,97],[89,89],color=BLUE,lw=1)

# Top metric cards — fixed spacing (taller cards, separated text)
cards=[('3+ categories dominated','83%','win rate'),('All 4 dominated','95%','win rate'),
       ('3-pointers assisted','83%','via passing'),('Model accuracy','79%','AUC 0.87')]
cw=22; gap=2; x0=3
for i,(lab,val,sub) in enumerate(cards):
    x=x0+i*(cw+gap)
    ax.add_patch(FancyBboxPatch((x,72),cw,14,boxstyle="round,pad=0.3,rounding_size=1.2",
                 facecolor=LGRAY,edgecolor='none'))
    ax.text(x+cw/2,84,lab,fontsize=7.5,color=GRAY,ha='center',va='center')
    ax.text(x+cw/2,79.5,val,fontsize=17,fontweight='bold',color=DARK,ha='center',va='center')
    ax.text(x+cw/2,74.5,sub,fontsize=7.5,color=GRAY,ha='center',va='center')

ax.text(3,68,'Tactical priorities (by predictive importance)',fontsize=11,fontweight='bold',color=DARK,va='top')
priorities=[('1','Win the assist battle','Strongest single lever — 1.19x win odds per assist',44,BLUE),
            ('2','Win the rebound battle','With assists together: 94% win rate',31,'#378ADD'),
            ('3','Steals & blocks','Complementary factors',23,'#85B7EB')]
y=61
for num,title,desc,pct,col in priorities:
    ax.add_patch(Rectangle((3,y-4.5),94,8,facecolor='white',edgecolor='#D3D1C7',lw=0.5))
    ax.text(5.5,y,num,fontsize=12,fontweight='bold',color=BLUE,va='center')
    ax.text(10,y+1.4,title,fontsize=10,fontweight='bold',color=DARK,va='center')
    ax.text(10,y-1.9,desc,fontsize=8,color=GRAY,va='center')
    bx=68; bw=20
    ax.add_patch(Rectangle((bx,y-1),bw,2,facecolor=LGRAY,edgecolor='none'))
    ax.add_patch(Rectangle((bx,y-1),bw*pct/44,2,facecolor=col,edgecolor='none'))
    ax.text(bx+bw+1.5,y,f'{pct}%',fontsize=9,fontweight='bold',color=DARK,va='center')
    y-=9.3

# Two insight boxes — taller to fit 3 lines
ax.add_patch(FancyBboxPatch((3,15),45.5,14,boxstyle="round,pad=0.3,rounding_size=1",facecolor='white',edgecolor='#D3D1C7',lw=0.5))
ax.text(5.5,27,'Passing playbook',fontsize=10,fontweight='bold',color=DARK,va='top')
ax.text(5.5,23,'• 9+ players assisting gives 56% win rate\n• Guards create 57% of all assists\n• Share the ball; 3-pointers need passing',fontsize=8,color='#444441',va='top',linespacing=1.6)

ax.add_patch(FancyBboxPatch((51.5,15),45.5,14,boxstyle="round,pad=0.3,rounding_size=1",facecolor='white',edgecolor='#D3D1C7',lw=0.5))
ax.text(54,27,'Momentum management',fontsize=10,fontweight='bold',color=DARK,va='top')
ax.text(54,23,'• 3rd quarter most decisive (r=0.47)\n• Down 10+ at half gives 14% comeback\n• Level entering Q4: winner takes 90%',fontsize=8,color='#444441',va='top',linespacing=1.6)

ax.add_patch(FancyBboxPatch((3,3),94,9,boxstyle="round,pad=0.3,rounding_size=1",facecolor='#E6F1FB',edgecolor='none'))
ax.text(50,7.5,'In one sentence: dominate three of four categories — assists above all — share the ball widely,\nand protect the third quarter.',fontsize=9.5,color=NAVY,ha='center',va='center',linespacing=1.5,style='italic')

plt.savefig('/home/claude/figures/fig10_dashboard.png',facecolor='white')
plt.close()
print("fixed")
