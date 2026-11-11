"""Analisis reproducible de diesel. Ejecutar desde cualquier directorio.
Salidas: datos mensuales, controles, validacion, predicciones, metricas y figuras.
El Excel original nunca se modifica. No se descarga informacion al ejecutarlo.
"""
from pathlib import Path
import os, sys, json, re, hashlib, platform, warnings, argparse
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL','2')
os.environ.setdefault('TF_ENABLE_ONEDNN_OPTS','0')
os.environ.setdefault('OMP_NUM_THREADS','2')
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import acf
import statsmodels, sklearn, tensorflow as tf
from tensorflow import keras
tf.config.threading.set_inter_op_parallelism_threads(2)
tf.config.threading.set_intra_op_parallelism_threads(2)
tf.config.experimental.enable_op_determinism()

BASE=Path(__file__).resolve().parents[1]
OUT=BASE/'resultados'; FIG=OUT/'figuras'; MODELS=OUT/'modelos'
MONTHS={m:i+1 for i,m in enumerate('enero febrero marzo abril mayo junio julio agosto septiembre octubre noviembre diciembre'.split())}
COLS=['GASOLINA_AUTOMOTOR','GASOLINA_AVIACION','GAS_LICUADO','KEROSENO','JET_FUEL','DIESEL','FUEL_OIL','ACEITES_AUTOMOTORES','ACEITES_INDUSTRIALES','GRASAS','ASFALTO','PARAFINA']
UNITS={c:('toneladas metricas' if c in COLS[-3:] else 'miles de barriles') for c in COLS}
WINDOW=12

def load_data(path):
    raw=pd.read_excel(path,sheet_name='H04',header=None)
    records=[];years=[];year=None;pre=False
    for i,row in raw.iterrows():
        v=str(row.iloc[1]).strip();match=re.fullmatch(r'(\d{4})(?:\s*\(p\))?',v,re.I)
        if match:
            year=int(match[1]);pre='(p)' in v.lower()
            years.append({'ANIO':year,'DIESEL_PUBLICADO':float(row.iloc[7])});continue
        month=MONTHS.get(v.lower())
        if month and year:
            rec={'FECHA':pd.Timestamp(year,month,1),'ANIO':year,'MES':month,'PRELIMINAR':pre,'FILA_EXCEL':i+1}
            for c,val in zip(COLS,row.iloc[2:14]):rec[c]=pd.to_numeric(val,errors='coerce')
            records.append(rec)
    df=pd.DataFrame(records).sort_values('FECHA').reset_index(drop=True)
    assert not df.FECHA.duplicated().any(),'Fechas duplicadas'
    assert pd.date_range(df.FECHA.min(),df.FECHA.max(),freq='MS').equals(pd.DatetimeIndex(df.FECHA)),'Meses ausentes'
    assert df.DIESEL.notna().all() and np.isfinite(df.DIESEL).all(),'Diesel incompleto'
    assert (df.DIESEL>0).all(),'Valores no positivos de diesel'
    # NO sumar barriles con toneladas. Los siete carburantes si comparten unidad.
    df['TOTAL_CARBURANTES_MILES_BARRILES']=df[COLS[:7]].sum(axis=1,min_count=7)
    annual=df.groupby('ANIO').agg(DIESEL_SUMA=('DIESEL','sum'),MESES=('MES','size'),PRELIMINAR=('PRELIMINAR','max')).reset_index()
    annual=annual.merge(pd.DataFrame(years),on='ANIO');annual['DIFERENCIA']=annual.DIESEL_SUMA-annual.DIESEL_PUBLICADO
    assert annual.DIFERENCIA.abs().max()<0.001,'No coincide la suma con los acumulados'
    return df,annual

def metrics(actual,pred):
    y=np.asarray(actual).ravel();p=np.asarray(pred).ravel()
    assert len(y)==len(p) and np.isfinite(p).all()
    e=y-p
    return {'N':len(y),'MAE':float(np.abs(e).mean()),'RMSE':float(np.sqrt((e**2).mean())),
            'MAPE':float(100*np.mean(np.abs(e/y))),'SESGO_REAL_MENOS_PRED':float(e.mean()),'DESV_ERROR':float(e.std(ddof=1))}

def arima_fit(y,order):
    with warnings.catch_warnings(record=True) as ww:
        warnings.simplefilter('always')
        fit=ARIMA(np.asarray(y),order=order).fit()
    return fit,[str(w.message) for w in ww]

def arima_rolling(fit,actual):
    pred=[];state=fit
    for y in actual:
        pred.append(float(np.asarray(state.forecast(1))[0]));state=state.extend([float(y)])
    return np.array(pred)

def windows(scaled,start,end):
    x=np.array([scaled[t-WINDOW:t] for t in range(start,end)],dtype=np.float32)
    y=scaled[start:end].astype(np.float32)
    return x,y

def lstm_fit(y,units,activation,epochs,seed=42,shuffle=False):
    keras.backend.clear_session();keras.utils.set_random_seed(seed)
    scaler=MinMaxScaler();z=scaler.fit_transform(np.asarray(y).reshape(-1,1))
    x,target=windows(z,WINDOW,len(z))
    model=keras.Sequential([keras.Input(shape=(WINDOW,1)),keras.layers.LSTM(units,activation=activation),keras.layers.Dense(1)])
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001),loss='mse')
    history=model.fit(x,target,epochs=epochs,batch_size=32,shuffle=shuffle,verbose=0)
    return model,scaler,history.history['loss']

def lstm_rolling(model,scaler,all_y,start,end):
    z=scaler.transform(np.asarray(all_y).reshape(-1,1))
    x,_=windows(z,start,end)
    return scaler.inverse_transform(model(x,training=False).numpy()).ravel()

def savefig(name):
    plt.tight_layout();plt.savefig(FIG/name,dpi=190,bbox_inches='tight');plt.close()

def main():
    for p in [OUT,FIG,MODELS,BASE/'datos/procesados']:p.mkdir(parents=True,exist_ok=True)
    source=next((BASE/'datos/originales').glob('*.xlsx'));df,annual=load_data(source)
    y=df.DIESEL.to_numpy();dates=df.FECHA;n=len(y);cut=int(n*0.8);validation_start=cut-60
    assert n==416 and cut==332 and validation_start==272
    df.to_csv(BASE/'datos/procesados/produccion_mensual.csv',index=False,encoding='utf-8-sig')
    annual.to_csv(OUT/'controles_anuales.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame([{'VARIABLE':c,'UNIDAD':UNITS[c],'FALTANTES':int(df[c].isna().sum()),'CEROS':int((df[c]==0).sum())} for c in COLS]).to_csv(OUT/'diccionario_y_calidad.csv',index=False,encoding='utf-8-sig')
    partitions=[('Entrenamiento interno',0,validation_start),('Validacion temporal',validation_start,cut),('Desarrollo completo',0,cut),('Prueba',cut,n)]
    pd.DataFrame([{'BLOQUE':name,'INICIO':str(dates.iloc[a].date()),'FIN':str(dates.iloc[b-1].date()),'N':b-a} for name,a,b in partitions]).to_csv(OUT/'particiones.csv',index=False)
    print('Datos auditados: 416 meses. Busqueda temporal.',flush=True)
    search=[];fits={};orders=[(0,1,0),(1,1,0),(0,1,1),(1,1,1),(2,1,2)]
    for order in orders:
        fit,msg=arima_fit(y[:validation_start],order)
        pred=arima_rolling(fit,y[validation_start:cut]);m=metrics(y[validation_start:cut],pred)
        search.append({'FAMILIA':'ARIMA','CONFIG':str(order),'CONVERGE':bool(fit.mle_retvals.get('converged',True)),**m,'AVISOS':' | '.join(msg)})
    best_arima=min([r for r in search if r['CONVERGE']],key=lambda r:r['RMSE'])
    import ast
    order=ast.literal_eval(best_arima['CONFIG'])
    configs=[(16,'tanh',20),(50,'relu',20),(16,'tanh',60),(50,'relu',60)]
    for units,act,epochs in configs:
        print('Validando LSTM',units,act,epochs,flush=True)
        model,scaler,loss=lstm_fit(y[:validation_start],units,act,epochs)
        pred=lstm_rolling(model,scaler,y,validation_start,cut)
        m=metrics(y[validation_start:cut],pred)
        search.append({'FAMILIA':'LSTM','CONFIG':f'{units},{act},{epochs}','CONVERGE':True,**m,'AVISOS':''})
    best_lstm=min([r for r in search if r['FAMILIA']=='LSTM'],key=lambda r:r['RMSE'])
    units,act,epochs=best_lstm['CONFIG'].split(',');units=int(units);epochs=int(epochs)
    pd.DataFrame(search).to_csv(OUT/'validacion_configuraciones.csv',index=False,encoding='utf-8-sig')
    # Las referencias simples tambien se seleccionan/evaluan en validacion.
    val_baselines=[{'MODELO':'Ingenuo',**metrics(y[validation_start:cut],y[validation_start-1:cut-1])},
                   {'MODELO':'Estacional',**metrics(y[validation_start:cut],y[validation_start-12:cut-12])},
                   {'MODELO':'ARIMA',**{k:best_arima[k] for k in metrics([1,2],[1,2])}},
                   {'MODELO':'LSTM',**{k:best_lstm[k] for k in metrics([1,2],[1,2])}}]
    pd.DataFrame(val_baselines).to_csv(OUT/'validacion_familias.csv',index=False)
    selected_family=min(val_baselines,key=lambda r:r['RMSE'])['MODELO']
    final_arima,msg=arima_fit(y[:cut],order)
    if not final_arima.mle_retvals.get('converged',True):raise RuntimeError('ARIMA final no converge')
    final_arima.save(MODELS/'arima.pkl')
    pred=pd.DataFrame({'FECHA':dates.iloc[cut:].to_numpy(),'REAL':y[cut:],
                       'INGENUO':y[cut-1:n-1],'ESTACIONAL':y[cut-12:n-12],
                       'ARIMA':arima_rolling(final_arima,y[cut:])})
    sensitivity=[];model=scaler=loss=None
    for seed in [42,43,44]:
        print('Ajuste final LSTM semilla',seed,flush=True)
        mm,ss,ll=lstm_fit(y[:cut],units,act,epochs,seed=seed)
        pp=lstm_rolling(mm,ss,y,cut,n)
        sensitivity.append({'SEMILLA':seed,**metrics(y[cut:],pp)})
        if seed==42:
            model,scaler,loss=mm,ss,ll;pred['LSTM']=pp;model.save(MODELS/'lstm.keras')
            (MODELS/'escalador.json').write_text(json.dumps({'min':float(ss.data_min_[0]),'max':float(ss.data_max_[0]),'feature_range':[0,1]},indent=2))
    pd.DataFrame(sensitivity).to_csv(OUT/'sensibilidad_semillas.csv',index=False)
    pred.to_csv(OUT/'predicciones_prueba.csv',index=False)
    met=pd.DataFrame([{'MODELO':c,**metrics(pred.REAL,pred[c])} for c in pred.columns[2:]])
    met.to_csv(OUT/'metricas_prueba.csv',index=False)
    byyear=[]
    for year,g in pred.groupby(pred.FECHA.dt.year):
        for c in pred.columns[2:]:byyear.append({'ANIO':year,'MODELO':c,**metrics(g.REAL,g[c])})
    pd.DataFrame(byyear).to_csv(OUT/'metricas_por_anio.csv',index=False)
    # Diagnostico del protocolo original: NO se usa para seleccionar ni concluir.
    basearima,_=arima_fit(y[:cut],(2,1,2))
    fixed=basearima.forecast(n-cut)
    bm,bs,_=lstm_fit(y[:cut],50,'relu',20,seed=42,shuffle=True)
    bp=lstm_rolling(bm,bs,y,cut+12,n)
    pd.DataFrame([{'MODELO':'ARIMA_ORIGINAL_84_MULTIPASO',**metrics(y[cut:],fixed)},
                  {'MODELO':'LSTM_ORIGINAL_72_UN_PASO_SEMILLA_42',**metrics(y[cut+12:],bp)}]).to_csv(OUT/'diagnostico_protocolo_original.csv',index=False)
    # Sensibilidad multihorizonte: siete origenes, mismos parametros, 12 meses recursivos.
    records=[];state=final_arima
    for t in range(cut,n):
        if (t-cut)%12==0 and t+12<=n:
            pa=np.asarray(state.forecast(12));hist=list(y[:t]);last=y[t-1];season=y[t-12:t]
            for h in range(1,13):
                z=scaler.transform(np.asarray(hist[-12:]).reshape(-1,1)).astype('float32')[None,:,:]
                pl=float(scaler.inverse_transform(model(z,training=False).numpy())[0,0]);hist.append(pl)
                records.append({'ORIGEN':str(dates.iloc[t-1].date()),'FECHA':str(dates.iloc[t+h-1].date()),'H':h,'REAL':y[t+h-1],
                                'INGENUO':last,'ESTACIONAL':season[h-1],'ARIMA':pa[h-1],'LSTM':pl})
        state=state.extend([y[t]])
    mult=pd.DataFrame(records);mult.to_csv(OUT/'predicciones_multihorizonte.csv',index=False)
    hm=[]
    for h,g in mult.groupby('H'):
        for c in ['INGENUO','ESTACIONAL','ARIMA','LSTM']:hm.append({'H':h,'MODELO':c,**metrics(g.REAL,g[c])})
    pd.DataFrame(hm).to_csv(OUT/'metricas_horizonte.csv',index=False)
    # Descriptivos: ningun cambio causal se infiere de esta serie univariada.
    stats={'n':n,'inicio':str(dates.iloc[0].date()),'fin':str(dates.iloc[-1].date()),'train_n':cut,'test_n':n-cut,
           'val_n':60,'train_inicio':str(dates.iloc[0].date()),'train_fin':str(dates.iloc[cut-1].date()),
           'test_inicio':str(dates.iloc[cut].date()),'test_fin':str(dates.iloc[-1].date()),
           'media':float(y.mean()),'mediana':float(np.median(y)),'sd':float(y.std(ddof=1)),
           'min':float(y.min()),'min_fecha':str(dates.iloc[y.argmin()].date()),'max':float(y.max()),'max_fecha':str(dates.iloc[y.argmax()].date()),
           'preliminares_n':int(df.PRELIMINAR.sum()),'preliminares_anios':df.loc[df.PRELIMINAR,'ANIO'].unique().tolist(),
           'faltantes_diesel':int(df.DIESEL.isna().sum()),'duplicados':int(df.FECHA.duplicated().sum()),
           'max_diff_anual':float(annual.DIFERENCIA.abs().max()),'arima_order':order,'lstm_units':units,'lstm_activation':act,'lstm_epochs':epochs,
           'lstm_params':int(model.count_params()),'seleccion_validacion':selected_family,'semilla_principal':42,
           'anual_max':annual.loc[annual.MESES.eq(12)].sort_values('DIESEL_SUMA').iloc[-1].to_dict(),
           'variacion_ene_ago_2024_2023':float((df.loc[(df.ANIO==2024)&(df.MES<=8),'DIESEL'].sum()/df.loc[(df.ANIO==2023)&(df.MES<=8),'DIESEL'].sum()-1)*100),
           'versiones':{'Python':platform.python_version(),'numpy':np.__version__,'pandas':pd.__version__,'statsmodels':statsmodels.__version__,'sklearn':sklearn.__version__,'tensorflow':tf.__version__,'keras':keras.__version__,'matplotlib':matplotlib.__version__},
           'sha256_excel':hashlib.sha256(source.read_bytes()).hexdigest(),'avisos_arima_final':msg}
    (OUT/'resumen.json').write_text(json.dumps(stats,indent=2,ensure_ascii=False,default=lambda o:o.item()),encoding='utf-8')
    (OUT/'perdida_entrenamiento.json').write_text(json.dumps(loss))
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    plt.figure(figsize=(9,3.5));plt.plot(dates,y,color='#235789',lw=1.3);plt.axvline(dates.iloc[cut],ls='--',color='#a43a38',label='Inicio de prueba');plt.ylabel('Miles de barriles por mes');plt.legend();savefig('01_serie_mensual.png')
    plt.figure(figsize=(9,3.5));colors=['#c27b27' if p else '#235789' for p in annual.PRELIMINAR];plt.bar(annual.ANIO,annual.DIESEL_SUMA,color=colors);plt.ylabel('Miles de barriles acumulados');plt.xlabel('Año; 2024 incluye solo enero–agosto');savefig('02_acumulados_anuales.png')
    plt.figure(figsize=(8,3.5));plt.boxplot([y[:cut][df.MES.iloc[:cut].to_numpy()==m] for m in range(1,13)],tick_labels=list(range(1,13)));plt.xlabel('Mes; bloque de desarrollo');plt.ylabel('Miles de barriles por mes');savefig('03_distribucion_mensual.png')
    fig,axs=plt.subplots(1,2,figsize=(9,3.2));axs[0].stem(np.arange(25),acf(y[:cut],nlags=24));axs[1].stem(np.arange(25),acf(np.diff(y[:cut]),nlags=24));axs[0].set_title('Nivel');axs[1].set_title('Primera diferencia')
    for ax in axs:ax.set_xlabel('Rezago mensual');ax.set_ylim(-1,1.05)
    savefig('04_autocorrelacion.png')
    plt.figure(figsize=(9,4));plt.plot(pred.FECHA,pred.REAL,color='black',lw=1.5,label='Observado')
    for c,color in [('INGENUO','#909090'),('ARIMA','#235789'),('LSTM','#c27b27')]:plt.plot(pred.FECHA,pred[c],label=c,color=color,alpha=.85,lw=1)
    plt.ylabel('Miles de barriles por mes');plt.legend(ncol=4);savefig('05_predicciones_prueba.png')
    plt.figure(figsize=(8,3.4));xx=np.arange(len(met));plt.bar(xx-.18,met.MAE,.36,label='MAE',color='#235789');plt.bar(xx+.18,met.RMSE,.36,label='RMSE',color='#c27b27');plt.xticks(xx,met.MODELO);plt.ylabel('Miles de barriles');plt.legend();savefig('06_metricas.png')
    fig,axs=plt.subplots(2,1,figsize=(9,5),sharex=True)
    for ax,c in zip(axs,['ARIMA','LSTM']):ax.plot(pred.FECHA,pred.REAL-pred[c],lw=1);ax.axhline(0,color='black',ls='--');ax.set_ylabel(c+'\nError')
    savefig('07_errores_tiempo.png')
    plt.figure(figsize=(8,3.5));plt.boxplot([pred.REAL-pred[c] for c in pred.columns[2:]],tick_labels=pred.columns[2:]);plt.axhline(0,color='gray',ls='--');plt.ylabel('Error real menos predicho; miles de barriles');savefig('08_distribucion_errores.png')
    plt.figure(figsize=(8,3));plt.plot(range(1,len(loss)+1),loss,color='#235789');plt.xlabel('Época');plt.ylabel('MSE en escala normalizada');savefig('09_entrenamiento_lstm.png')
    hdf=pd.DataFrame(hm);plt.figure(figsize=(8,3.5))
    for c,g in hdf.groupby('MODELO'):plt.plot(g.H,g.RMSE,marker='o',label=c)
    plt.xlabel('Horizonte en meses; siete orígenes');plt.ylabel('RMSE; miles de barriles');plt.legend();savefig('10_horizonte.png')
    sample=df[['FECHA','DIESEL','PRELIMINAR','FILA_EXCEL']].tail(8).copy();sample.FECHA=sample.FECHA.dt.strftime('%Y-%m');sample.DIESEL=sample.DIESEL.round(3)
    fig,ax=plt.subplots(figsize=(8,2.8));ax.axis('off');table=ax.table(cellText=sample.values,colLabels=sample.columns,loc='center',cellLoc='center');table.auto_set_font_size(False);table.set_fontsize(10);table.scale(1,1.5);savefig('11_muestra_datos.png')
    print(met.to_string(index=False),flush=True);print('FINALIZADO',flush=True)

if __name__=='__main__':main()
