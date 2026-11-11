# Producción mensual de diésel en Bolivia

Autor: Andre Vladimir Arroyo Hinojosa.

Este paquete acompaña la tesis y permite reproducir la extracción, la auditoría y la comparación retrospectiva de ARIMA y LSTM sobre 416 meses, de enero de 1990 a agosto de 2024. La variable se expresa en miles de barriles mensuales. Los registros de 2022, 2023 y 2024 son preliminares; 2024 tiene ocho meses. No se generan pronósticos operativos posteriores al cierre de la base.

## Resultados principales

La prueba incluye 84 fechas idénticas para cuatro modelos. ARIMA(1,1,1) obtuvo RMSE 41,5262; LSTM de 50 unidades ReLU y 60 épocas obtuvo 50,4432; la referencia ingenua obtuvo 42,6366 y la estacional 85,9733. MAE y RMSE se expresan en miles de barriles; MAPE es un porcentaje. Las configuraciones se eligieron por validación anterior a la prueba. Se incluyen tres semillas LSTM y siete orígenes con doce horizontes como análisis complementarios.

La ventaja de ARIMA corresponde a esta serie y este protocolo; no demuestra superioridad universal ni significación estadística. La referencia ingenua es competitiva.

## Estructura

- `datos/originales/`: Excel íntegro, hoja H04, con encabezados y notas.
- `datos/procesados/produccion_mensual.csv`: 416 registros; contiene productos de unidades diferentes. Consultar el diccionario antes de agregarlos.
- `codigo/analisis_diesel.py`: análisis ejecutable con rutas relativas al propio archivo.
- `codigo/verificar_resultados.py`: comprobación independiente de fechas, predicciones de referencia, métricas, horizontes y acumulados anuales.
- `codigo/original/diplomado_colab_original.py`: programa original conservado para trazabilidad; sus órdenes de Colab no se ejecutan como un script local ordinario.
- `resultados/`: predicciones, métricas, controles, configuraciones, sensibilidad, modelos y figuras.
- `requirements.txt`: versiones del entorno analítico.
- `MANIFIESTO_SHA256.json`: huellas de los archivos del paquete, excepto el manifiesto mismo.

## Ejecución en Windows

Se utilizó Python 3.12.14 y entrenamiento en CPU. Desde la carpeta del paquete:

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python codigo/analisis_diesel.py
.venv\Scripts\python codigo/verificar_resultados.py
.venv\Scripts\python codigo/figuras_adicionales.py
.venv\Scripts\python codigo/complementos_documentales.py
```

La instalación necesita internet y espacio suficiente para TensorFlow y sus dependencias. El análisis utiliza el Excel local y no descarga nuevos datos. Las salidas se regeneran y sustituyen; el Excel original se conserva. La duración depende del equipo. Determinismo y semillas reducen la variabilidad, pero versiones, hardware y bibliotecas numéricas pueden producir pequeñas diferencias.

Los modelos se ajustan con 332 meses hasta agosto de 2017. La selección interna usa 272 meses y 60 meses de validación. La prueba abarca septiembre de 2017 a agosto de 2024. En la evaluación principal se incorporan observaciones reales anteriores sin reestimar pesos o parámetros. En cada origen del análisis multihorizonte se predicen doce meses sin utilizar observaciones posteriores a ese origen.

## Diferencias respecto del programa de partida

El programa original comparaba 84 pasos ARIMA desde un origen fijo con 72 predicciones LSTM basadas en observaciones previas. El protocolo corregido evalúa las mismas 84 fechas con el mismo acceso a información histórica. Las métricas de `diagnostico_protocolo_original.csv` documentan la limitación de la comparación inicial. No deben confundirse con `metricas_prueba.csv`. El entrenamiento original no fijaba semilla; su red no puede reconstruirse exactamente solo con el código recibido.

Se eliminó la suma de barriles con toneladas métricas. `TOTAL_CARBURANTES` agrega únicamente siete carburantes de la misma unidad y no se utiliza como predictor de diésel. Los totales anuales del cuadro se usan como control, nunca como observaciones adicionales.

## Comprobación y entrega

La verificación independiente pasó para los resultados incluidos. `resumen.json` registra el entorno y el hash del Excel. Los archivos `.pkl` y `.keras` son resultados propios de este proyecto; requieren bibliotecas compatibles para cargarse. No deben cargarse archivos de modelos de origen desconocido.

El mapa utilizado en la tesis pulida procede de Naciones Unidas, mapa n.º 3875, revisión 2, enero de 2004. El PDF original se conserva en datos/fuentes_cartograficas/ y su imagen en resultados/figuras/mapa_bolivia_onu_2004.png. Solo ubica el área nacional: no representa refinerías ni datos de producción actuales. La imagen anterior mapa_contexto.png se conserva como respaldo del material recibido y no se utiliza en la tesis pulida. Los demás gráficos proceden del análisis. complementos_documentales.py genera estadísticas descriptivas adicionales y una captura del código entregado; no vuelve a entrenar los modelos.

Repositorio de entrega: https://github.com/AndreVA21/prediccion-diesel-bolivia . Los datos están en [datos/](datos/) y los programas en [codigo/](codigo/). Para el CD solicitado por la plantilla, descargar el repositorio y conservar esta estructura. Los integrantes definitivos del tribunal y la grabación física del CD siguen pendientes de gestión del autor. Las cartas firmadas no forman parte de este repositorio público.
