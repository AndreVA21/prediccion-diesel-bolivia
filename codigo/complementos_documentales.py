"""Generar controles numéricos adicionales y una captura del código entregado.
No modifica modelos, particiones ni predicciones.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
BASE=Path(__file__).resolve().parents[1]
R=BASE/'resultados';FIG=R/'figuras'
data=pd.read_csv(BASE/'datos/procesados/produccion_mensual.csv')
dev=data.iloc[:332]
monthly=dev.groupby('MES').DIESEL.agg(N='size',MEDIANA='median',MEDIA='mean')
monthly['Q1']=dev.groupby('MES').DIESEL.quantile(.25)
monthly['Q3']=dev.groupby('MES').DIESEL.quantile(.75)
monthly.to_csv(R/'descripcion_por_mes.csv',float_format='%.10f')
def acf_manual(values,k):
    centered=np.asarray(values,dtype=float)-np.mean(values)
    return float(np.dot(centered[k:],centered[:-k])/np.dot(centered,centered)) if k else 1.
y=dev.DIESEL.to_numpy()
acf=pd.DataFrame({'REZAGO':range(25),'NIVEL':[acf_manual(y,k) for k in range(25)],'PRIMERA_DIFERENCIA':[acf_manual(np.diff(y),k) for k in range(25)]})
acf.to_csv(R/'autocorrelacion_descriptiva.csv',index=False,float_format='%.10f')
pred=pd.read_csv(R/'predicciones_prueba.csv')
rows=[]
for name in ['INGENUO','ESTACIONAL','ARIMA','LSTM']:
    err=pred.REAL-pred[name]
    maxidx=err.abs().idxmax()
    rows.append({'MODELO':name,'MEDIANA_ERROR':float(err.median()),'Q1':float(err.quantile(.25)),'Q3':float(err.quantile(.75)),'SOBREESTIMACIONES':int((err<0).sum()),'SUBESTIMACIONES':int((err>0).sum()),'MAX_ERROR_ABSOLUTO':float(err.abs().max()),'FECHA_MAX_ERROR':pred.loc[maxidx,'FECHA']})
pd.DataFrame(rows).to_csv(R/'descripcion_errores.csv',index=False,float_format='%.10f')
loss=json.loads((R/'perdida_entrenamiento.json').read_text())
summary={'mediana_mayor_mes':int(monthly.MEDIANA.idxmax()),'mediana_menor_mes':int(monthly.MEDIANA.idxmin()),'mediana_mayor':float(monthly.MEDIANA.max()),'mediana_menor':float(monthly.MEDIANA.min()),'mse_epoca_1':loss[0],'mse_epoca_60':loss[-1],'reduccion_mse_ajuste_pct':100*(1-loss[-1]/loss[0])}
(R/'complementos_descriptivos.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
# Imagen legible de un fragmento real, sin presentar código inventado.
lines=(BASE/'codigo/analisis_diesel.py').read_text(encoding='utf-8').splitlines()
start=next(i for i,s in enumerate(lines) if s.startswith('def arima_rolling('))
end=next(i for i in range(start+1,len(lines)) if lines[i].startswith('def '))
snippet=lines[start:end]
font_path=Path('C:/Windows/Fonts/consola.ttf')
font=ImageFont.truetype(str(font_path),24) if font_path.exists() else ImageFont.load_default(size=24)
im=Image.new('RGB',(1580,90+40*len(snippet)),'white');draw=ImageDraw.Draw(im)
draw.text((24,15),'analisis_diesel.py  |  fragmento del archivo entregado',font=font,fill='#111111')
for j,line in enumerate(snippet):draw.text((24,65+j*40),f'{start+j+1:3}  {line}',font=font,fill='#111111')
im.save(FIG/'15_captura_codigo.png')
print(json.dumps(summary,ensure_ascii=False))
