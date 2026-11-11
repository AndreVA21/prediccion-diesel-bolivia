"""Comprobaciones independientes de las tablas de resultados, sin entrenar."""
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
data=pd.read_csv(ROOT/'datos/procesados/produccion_mensual.csv')
p=pd.read_csv(ROOT/'resultados/predicciones_prueba.csv')
m=pd.read_csv(ROOT/'resultados/metricas_prueba.csv').set_index('MODELO')
assert len(data)==416 and len(p)==84
assert p.FECHA.tolist()==data.FECHA.iloc[332:].tolist()
np.testing.assert_allclose(p.REAL,data.DIESEL.iloc[332:])
np.testing.assert_allclose(p.INGENUO,data.DIESEL.iloc[331:-1])
np.testing.assert_allclose(p.ESTACIONAL,data.DIESEL.iloc[320:-12])
for name in m.index:
    error=p.REAL.to_numpy()-p[name].to_numpy()
    # CSV contiene predicciones float32 de la red; tolerancia muy inferior a 0,01.
    np.testing.assert_allclose(np.abs(error).mean(),m.loc[name,'MAE'],rtol=1e-7,atol=1e-6)
    np.testing.assert_allclose(np.sqrt(np.square(error).mean()),m.loc[name,'RMSE'],rtol=1e-7,atol=1e-6)
    np.testing.assert_allclose(100*np.mean(np.abs(error/p.REAL)),m.loc[name,'MAPE'],rtol=1e-7,atol=1e-6)
mult=pd.read_csv(ROOT/'resultados/predicciones_multihorizonte.csv')
assert len(mult)==84 and mult.ORIGEN.nunique()==7
for _,row in mult[mult.H==1].iterrows():
    pp=p[p.FECHA==row.FECHA].iloc[0]
    for name in m.index:np.testing.assert_allclose(row[name],pp[name],rtol=1e-5)
assert pd.read_csv(ROOT/'resultados/controles_anuales.csv').DIFERENCIA.abs().max()<.001
print('CORRECTO: fechas, 84 predicciones comunes, referencias, metricas, siete origenes y controles anuales.')
