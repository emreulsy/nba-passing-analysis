import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.titlesize':13,
    'axes.titleweight':'bold','axes.labelsize':11,'axes.spines.top':False,'axes.spines.right':False,
    'axes.grid':True,'grid.alpha':0.25,'grid.linewidth':0.6,'figure.dpi':150,'savefig.dpi':200,'savefig.bbox':'tight'})
GRAY='#888780'; FIG='/home/claude/figures/'

# ===== FIG 9: Subgroup discovery — discovered winning/losing conditions =====
fig,ax=plt.subplots(figsize=(7,4.2))
labels=['Much more assists\n+ much more rebounds','Much more assists\n+ home','Very high assist level','Much more rebounds\n+ home','Much fewer assists\n(vs opp)','Much fewer assists\n+ much fewer rebounds']
vals=[93.6,85.7,77.5,79.4,15.9,8.0]
colors=['#0F6E56' if v>=50 else '#C0392B' for v in vals]
bars=ax.barh(labels[::-1],vals[::-1],color=colors[::-1])
ax.axvline(50,color=GRAY,ls='--',lw=0.8)
ax.set_xlabel('Win rate (%)')
ax.set_title('Subgroup discovery: strongest winning and losing conditions',loc='left')
for b,v in zip(bars,vals[::-1]):
    ax.text(v+1.5 if v>50 else v-1.5,b.get_y()+b.get_height()/2,f'{v:.0f}%',
            va='center',ha='left' if v>50 else 'right',fontsize=10,fontweight='bold')
ax.set_xlim(0,100)
plt.savefig(FIG+'fig9_subgroup.png');plt.close()
print("fig9_subgroup.png created")
