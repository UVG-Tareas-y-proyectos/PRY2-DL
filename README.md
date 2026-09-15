# Proyecto 2 · Detección de patrones transaccionales

Diego Patzan · 23525<br>
Ihan Marroquin · 23108

Sistema académico de dos etapas para revisar secuencias por remitente: un autoencoder
GRU que aprende normalidad y un clasificador con atención que reutiliza el encoder.
Incluye una ablación con la misma arquitectura supervisada entrenada desde cero,
un notebook ejecutable, un reporte ejecutivo y una interfaz Streamlit.

## Datos y alcance

El enunciado propone PaySim como dataset principal. La secuenciación por `nameOrig`
en PaySim no es viable para este ejercicio porque la mayoría de remitentes aparecen
solo una vez. Por ello usamos **IBM AML HI-Small** como fuente principal de las
secuencias. Los datos son sintéticos, incluyen pagos de varios tipos y **no son
remesas Guatemala–Estados Unidos**. No se debe presentar la puntuación como prueba
de delito ni como rendimiento operativo en ese corredor.

Fuente original: [IBM AML-Data](https://github.com/IBM/AML-Data) y
[dataset IBM AML en Kaggle](https://www.kaggle.com/datasets/ealtman2019/ibm-transactions-for-anti-money-laundering-aml).
Para reproducir sin cuenta Kaggle se usa un
[espejo público HI-Small](https://huggingface.co/datasets/OsamaMIT/IBM-AML-HI-Small).
El archivo CSV completo se mantiene fuera de Git por tamaño y por los términos
de uso del conjunto original (CDLA-Sharing-1.0).

## Ejecución

Requiere Python 3.10 o posterior. Desde la raíz:

```bash
python -m pip install -r requirements-train.txt
python scripts/download_data.py
python -m src.train --sample-percent 10 --epochs-ae 3 --epochs-cls 4
python -m streamlit run app.py
```

La aplicación lee `artifacts/results.json` y `artifacts/test_cases.json`, derivados
del conjunto de prueba. También acepta un ID de remitente de las ventanas de prueba
incluidas. No acepta perfiles ajenos al conjunto de prueba.

Para desplegar en Streamlit Community Cloud, seleccione este repositorio, la rama
`main` y `app.py` como archivo principal. Cloud instala `requirements.txt`, que
contiene solo las dependencias de la interfaz. Configure la app como pública para
que el evaluador abra el enlace sin cuenta; registre ese enlace en `mvp_url.txt`.

Abra `notebooks/proyecto2.ipynb` y ejecute todas las celdas. El notebook comprueba
si existe el CSV y lo descarga si hace falta. En Colab, sitúe primero el repositorio
en `/content` y abra el notebook desde esa carpeta. La ejecución con GPU T4 se diseñó
para menos de 30 minutos; el tiempo real depende de la red y del hardware.

## Diseño y control de filtraciones

- Se selecciona el 10 % de remitentes mediante un hash estable independiente de su
  etiqueta. No se sobremuestrean casos positivos para evaluación.
- Un segundo hash separa remitentes completos en 70 % entrenamiento, 15 % validación
  y 15 % prueba. Las ventanas cronológicas de un mismo remitente nunca cruzan splits.
- Cada ventana contiene hasta 24 transacciones consecutivas, sin solapamiento. Las
  ventanas de longitud uno se excluyen porque no expresan un patrón temporal.
- Las variables son monto transformado en logaritmo, intervalo de tiempo, hora
  cíclica, coincidencia de banco, diferencia de divisa, novedad del destino y formato
  de pago. No se utilizan ID, etiqueta, saldos posteriores ni información futura.
- `StandardScaler` se ajusta solo con las ventanas de entrenamiento. El autoencoder
  se entrena únicamente con ventanas normales. Las pérdidas ignoran padding.
- El clasificador transferido recibe pesos iniciales del encoder del autoencoder;
  el clasificador de ablación conserva arquitectura y épocas, pero inicia aleatorio.
- La validación selecciona umbrales por F2, calibra los logits y elige el peso de
  combinación. El conjunto de prueba permanece retenido hasta la evaluación final.
- La atención y el error de reconstrucción sirven para priorizar transacciones para
  revisión humana. No constituyen explicación causal.

## Archivos principales

| Archivo | Propósito |
| --- | --- |
| `src/data.py` | Muestreo, secuencias, features y separación por remitente |
| `src/models.py` | Autoencoder GRU y clasificador con atención |
| `src/train.py` | Entrenamiento, calibración, ablación y métricas |
| `notebooks/proyecto2.ipynb` | Exploración visual y experimento reproducible |
| `app.py` | MVP Streamlit |
| `artifacts/` | Resultados reales y ventanas de prueba del modelo entrenado |
| `report/reporte_ejecutivo.pdf` | Análisis para cumplimiento |

## Límites

El dataset es sintético, de otra geografía y otra mezcla de productos. El resultado
no mide desempeño en Guatemala. Las etiquetas a nivel de transacción se agregan a
ventana por OR; esto es adecuado para una alerta inicial, pero no identifica el
origen causal de la señal. La puntuación de etapa B se calibra en una muestra de
validación con pocos positivos. Antes de cualquier uso real se requiere revisión
jurídica, datos consentidos, evaluación temporal prospectiva, control de sesgo,
monitoreo de cambios y umbrales según capacidad del equipo de cumplimiento.
