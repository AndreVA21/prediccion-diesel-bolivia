from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
P=Path(__file__).resolve().parents[1]/'resultados'/'figuras'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10})
fig,ax=plt.subplots(figsize=(8,6));ax.set_xlim(0,10);ax.set_ylim(0,10);ax.axis('off')
blocks=[('Objetivo 1 · Auditar y describir','Excel H04 → 416 meses → unidades y controles'),('Objetivo 2 · Construir modelos','Validación temporal → ARIMA y LSTM → ajuste final'),('Objetivo 3 · Comparar predicciones','84 meses comunes → MAE, RMSE y MAPE'),('Objetivo 4 · Valorar la aplicación','Horizontes 1–12 meses → límites y decisión')]
for i,(a,b) in enumerate(blocks):
 y=8.1-i*2.15
 ax.add_patch(FancyBboxPatch((.5,y),9,1.5,boxstyle='round,pad=.1',facecolor='#edf2f7',edgecolor='#235789',linewidth=1.4))
 ax.text(5,y+1,a,ha='center',va='center',weight='bold');ax.text(5,y+.43,b,ha='center',va='center',fontsize=9.5)
 if i<3:ax.annotate('',xy=(5,y-.6),xytext=(5,y-.12),arrowprops=dict(arrowstyle='->',lw=1.5))
fig.tight_layout();fig.savefig(P/'12_flujograma.png',dpi=190);plt.close(fig)
fig,ax=plt.subplots(figsize=(8,2.5));ax.set_xlim(0,416);ax.set_ylim(0,2);ax.axis('off')
for start,width,color,label in [(0,272,'#235789','Ajuste interno\n272 meses'),(272,60,'#c27b27','Validación\n60'),(332,84,'#487b52','Prueba\n84')]:
 ax.barh(1,width,left=start,height=.7,color=color);ax.text(start+width/2,1,label,ha='center',va='center',color='white')
for x,text in [(0,'ene. 1990'),(272,'sep. 2012'),(332,'sep. 2017'),(416,'ago. 2024')]:ax.text(x,.45,text,ha='center',fontsize=8)
fig.tight_layout();fig.savefig(P/'13_particiones.png',dpi=190);plt.close(fig)
fig,ax=plt.subplots(figsize=(8,2.8));ax.axis('off')
text='Entradas para septiembre de 2017\nSeptiembre 2016, octubre 2016, …, agosto 2017\n12 observaciones anteriores → LSTM → septiembre 2017\n\nAl pasar a octubre se incorpora septiembre observado.\nEl valor de octubre nunca entra en su propia predicción.'
ax.text(.5,.5,text,ha='center',va='center',linespacing=1.8);fig.tight_layout();fig.savefig(P/'14_ventana.png',dpi=190);plt.close(fig)
