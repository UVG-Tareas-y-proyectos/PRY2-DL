"""Build the reproducible notebook; execution is a separate verification step."""
from pathlib import Path

import nbformat as nbf

root = Path(__file__).resolve().parents[1]
nb = nbf.v4.new_notebook()
nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
nb.metadata["language_info"] = {"name": "python"}


def md(text):
    nb.cells.append(nbf.v4.new_markdown_cell(text))


def code(text):
    nb.cells.append(nbf.v4.new_code_cell(text))


md("""# Proyecto 2 · Detección de patrones transaccionales con dos etapas

**Universidad del Valle de Guatemala · CC3092 Deep Learning**<br>
Diego Patzan · 23525<br>
Ihan Marroquin · 23108

**Datos sintéticos, no remesas guatemaltecas.** Este notebook usa IBM AML HI-Small para
construir secuencias por remitente. PaySim se descartó para la representación temporal:
sus identificadores de origen casi nunca se repiten, por lo que un historial por remitente
sería mayoritariamente de longitud uno. El archivo IBM se descarga de una copia pública
del conjunto original y se usa solo para investigación; no se redistribuye en el repositorio.

**Secuencia de trabajo:** muestreo y auditoría → visualización *antes* de entrenar →
autoencoder GRU solo con ventanas normales → clasificador con transferencia del encoder →
clasificador idéntico desde cero → ajuste de umbrales en validación → prueba retenida.
Las etiquetas y el ID del remitente no son entradas del modelo.
""")
md("""## 0. Preparación

En Google Colab con GPU T4, abra el notebook desde la raíz del repositorio y ejecute todas
las celdas. La primera ejecución descarga 475 MB de datos IBM. Si el repositorio es privado,
debe clonarse o subirse a Colab antes de abrir el notebook. El entrenamiento está limitado
a 10 % de remitentes, ventanas de 24 transacciones y pocas épocas para caber en 30 minutos.
""")
code("""from pathlib import Path
import os, sys, urllib.request, json
ROOT = Path.cwd()
if not (ROOT / 'src').exists() and (ROOT.parent / 'src').exists():
    ROOT = ROOT.parent
os.chdir(ROOT)
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
DATA = ROOT / 'data/raw/HI-Small_Trans.csv'
DATA.parent.mkdir(parents=True, exist_ok=True)
if not DATA.exists():
    url = 'https://huggingface.co/datasets/OsamaMIT/IBM-AML-HI-Small/resolve/main/HI-Small_Trans.csv?download=true'
    urllib.request.urlretrieve(url, DATA)
print('CSV disponible:', DATA, 'bytes:', DATA.stat().st_size)
""")
code("""import importlib.util, subprocess
needed = ('torch', 'sklearn', 'matplotlib', 'pandas')
if any(importlib.util.find_spec(name) is None for name in needed):
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-r',
                           str(ROOT / 'requirements-train.txt')])
print('Dependencias del experimento listas')
""")
md("""## 1. Ingeniería de secuencias y auditoría

Se muestrea un 10 % de remitentes por hash estable, **antes** de leer sus ventanas; así no
se sobreseleccionan positivos usando su etiqueta. El segundo hash asigna remitentes
completos a entrenamiento (70 %), validación (15 %) y prueba (15 %). Se ordenan por fecha
y se crean ventanas consecutivas no solapadas de hasta 24 movimientos; las ventanas de
una sola transacción se excluyen. Son features: logaritmo del monto, intervalo entre
operaciones, hora cíclica, banco igual/distinto, divisa igual/distinta, destino nuevo y
tipo de pago. El escalador se ajusta solo en entrenamiento. Una ventana es positiva si
contiene al menos una transacción marcada como lavado sintético.
""")
code("""from src.data import load_sample, build_windows, normalize
frame, raw_audit = load_sample(DATA, sample_percent=10)
windows, window_audit = build_windows(frame)
windows, scaler = normalize(windows)
print(json.dumps(raw_audit, indent=2))
print(json.dumps({k:v for k,v in window_audit.items() if k != 'lengths'}, indent=2))
print('No hay remitentes compartidos entre splits:',
      all(not ({w.sender for w in windows if w.split == a} &
               {w.sender for w in windows if w.split == b})
          for a,b in [('train','val'),('train','test'),('val','test')]))
""")
code("""import matplotlib.pyplot as plt
import numpy as np
lengths = window_audit['lengths']
fig, ax = plt.subplots(1, 2, figsize=(11, 3.5))
ax[0].hist(lengths, bins=range(2,26), color='#217c87')
ax[0].set(title='Longitud de secuencias', xlabel='Transacciones', ylabel='Ventanas')
splits = window_audit['split_counts']
ax[1].bar(list(splits), [splits[s]['positive']/splits[s]['total'] for s in splits], color='#ba5238')
ax[1].set(title='Proporción positiva por split', ylabel='Ventanas positivas / total')
plt.tight_layout(); plt.show()
""")
md("""### Ejemplos antes del modelado

Se muestran tres ventanas normales y tres sospechosas. El gráfico usa montos originales;
las etiquetas solo colorean para inspección y jamás entran al autoencoder.
""")
code("""examples = ([w for w in windows if w.split == 'train' and w.label == 0 and len(w.transactions) >= 4][:3]
            + [w for w in windows if w.split == 'train' and w.label == 1 and len(w.transactions) >= 4][:3])
fig, axes = plt.subplots(2, 3, figsize=(13, 6), sharey=False)
for ax, w in zip(axes.flat, examples):
    amounts = [tx['amount'] for tx in w.transactions]
    colors = ['#ba5238' if tx['transaction_label'] else '#217c87' for tx in w.transactions]
    ax.bar(range(1,len(amounts)+1), amounts, color=colors)
    ax.set_title(('Sospechosa' if w.label else 'Normal') + f' · {w.sender}')
    ax.set_xlabel('Orden temporal'); ax.set_ylabel('Monto pagado')
plt.tight_layout(); plt.show()
""")
md("""## 2. Dos etapas, transferencia y ablación

**Etapa A:** GRU encoder con estado comprimido y GRU decoder entrenados solo con ventanas
normales; la pérdida es el error cuadrático medio de posiciones reales (sin padding).
El error se convierte en percentil respecto al entrenamiento normal. El umbral se elige
en validación por F2, que prioriza recuperar casos sospechosos sin ignorar la carga
operativa de falsos positivos.

**Etapa B:** se copian los pesos del encoder al clasificador GRU con atención y se ajusta
todo el modelo (fine tuning). Se usa BCE con peso positivo igual a la raíz de la razón
negativos/positivos, limitado a 40; evita que la clase rara desaparezca sin generar
gradientes extremos. Los logits se calibran con validación. La ablación entrena el mismo
clasificador desde cero, con el mismo número de épocas. El peso de combinación A/B y los
umbrales se fijan en validación, nunca en prueba. La semilla 42 hace reproducible el
muestreo y la inicialización, aunque GPU puede introducir pequeñas diferencias.

**Nota curso:** la etapa A aplica representación auto supervisada y reconstrucción; la
etapa B aplica transferencia y fine tuning vistos en semana 9. La atención facilita
inspección, pero no prueba causalidad; debe contrastarse con error de reconstrucción y
revisión de cumplimiento.
""")
code("""from src.train import main
ART = ROOT / 'artifacts'
if not (ART / 'results.json').exists():
    results = main(str(DATA), str(ART), sample_percent=10, epochs_ae=3, epochs_cls=4)
else:
    results = json.loads((ART / 'results.json').read_text(encoding='utf-8'))
print('Entrenamiento y evaluación listos:', ART)
""")
code("""import pandas as pd
table = pd.DataFrame(results['test']).T[['average_precision','precision','recall','f2','alerts']]
display(table.style.format({'average_precision':'{:.4f}','precision':'{:.3f}',
                            'recall':'{:.3f}','f2':'{:.3f}'}))
print('Prevalencia positiva en prueba:', round(results['test_positive_rate'],5))
print('Peso de mezcla (A):', results['selection']['alpha'])
""")
md("""## 3. Trazabilidad de alertas y fallos

La atención señala movimientos que influyeron en el resumen del clasificador. El error
de reconstrucción señala operaciones que el modelo normal reprodujo peor. Ambos son
señales descriptivas; sin perturbación causal no deben presentarse como una causa
demostrada de la alerta. Las etiquetas de transacciones solo se usan aquí para revisión.
""")
code("""cases = json.loads((ART / 'test_cases.json').read_text(encoding='utf-8'))
tp = [c for c in cases if c['label'] and c['alert']]
fp = [c for c in cases if not c['label'] and c['alert']]
fn = [c for c in cases if c['label'] and not c['alert']]
chosen = tp[:3] + (fp + fn)[:2]
print('Casos disponibles: TP',len(tp),'FP',len(fp),'FN',len(fn))
for c in chosen:
    print('\\n', 'TP' if c['label'] and c['alert'] else ('FP' if c['alert'] else 'FN'),
          c['sender'], 'score',round(c['combined_score'],3))
    rank = np.argsort(np.asarray(c['attention']) + np.asarray(c['reconstruction_error']))[-3:][::-1]
    display(pd.DataFrame([dict(c['transactions'][i], position=int(i+1),
                               attention=c['attention'][i], error=c['reconstruction_error'][i])
                          for i in rank]))
""")
md("""## 4. Límites y reproducibilidad

IBM AML representa pagos bancarios sintéticos de otro contexto y no valida un sistema
regulatorio en Guatemala. La selección de 10 % mantiene la proporción de clase por
remitente aproximadamente, pero puede variar con la semilla. La etiqueta de una ventana
es positiva si **cualquier** transacción lo es; no se conoce la intención del cliente.
Los umbrales se optimizan para F2 académico, no para costos reales o capacidad de
investigación. La prueba se usó una sola vez al final. Para producción se requerirían
datos consentidos, revisión jurídica, calibración temporal, auditoría de sesgos y
validación prospectiva con especialistas. Véase el reporte ejecutivo para análisis.
""")

target = root / "notebooks/proyecto2.ipynb"
target.parent.mkdir(parents=True, exist_ok=True)
nbf.write(nb, target)
print(target)
