"""Create the 2,000–3,000-word executive PDF from measured results."""
from __future__ import annotations

import html
import json
import re
from pathlib import Path

import numpy as np
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (BaseDocTemplate, Frame, KeepTogether, NextPageTemplate, PageBreak,
                               PageTemplate, Paragraph, Spacer, Table, TableStyle)

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts"
results = json.loads((ART / "results.json").read_text(encoding="utf-8"))
cases = json.loads((ART / "test_cases.json").read_text(encoding="utf-8"))
target = ROOT / "report/reporte_ejecutivo.pdf"
target.parent.mkdir(parents=True, exist_ok=True)

windows_fonts = Path("C:/Windows/Fonts")
if (windows_fonts / "times.ttf").exists():
    regular_font, bold_font = windows_fonts / "times.ttf", windows_fonts / "timesbd.ttf"
else:
    linux_fonts = Path("/usr/share/fonts/truetype/liberation2")
    regular_font = linux_fonts / "LiberationSerif-Regular.ttf"
    bold_font = linux_fonts / "LiberationSerif-Bold.ttf"
pdfmetrics.registerFont(TTFont("TimesLocal", str(regular_font)))
pdfmetrics.registerFont(TTFont("TimesLocalBold", str(bold_font)))
pdfmetrics.registerFontFamily("TimesLocal", normal="TimesLocal", bold="TimesLocalBold")

styles = getSampleStyleSheet()
styles.add(ParagraphStyle(name="ReportH1", fontName="TimesLocalBold", fontSize=12,
                          leading=16, textColor=colors.black,
                          spaceBefore=15, spaceAfter=7, keepWithNext=True))
styles.add(ParagraphStyle(name="ReportBody", fontName="TimesLocal", fontSize=11.5,
                          leading=16, spaceAfter=9, alignment=TA_LEFT))
styles.add(ParagraphStyle(name="ReportSmall", fontName="TimesLocal", fontSize=9.3,
                          leading=12.5, spaceAfter=5, wordWrap="CJK"))

story = [NextPageTemplate("body"), PageBreak()]
word_parts = []


def paragraph(text, style="ReportBody"):
    word_parts.append(re.sub("<[^>]+>", " ", text))
    item = Paragraph(text, styles[style])
    story.append(KeepTogether([item]) if text.startswith("<b>Caso ") else item)


def heading(text):
    paragraph(text, "ReportH1")


heading("1. Resumen para dirección de cumplimiento")
paragraph("Una alerta útil debe señalar un historial de movimientos y permitir al analista abrir las operaciones que la motivaron. Construimos una demostración de dos etapas para esa tarea. La primera aprende a reconstruir patrones normales sin utilizar etiquetas de lavado; cuando una secuencia se aleja de lo aprendido, genera un puntaje de anomalía. La segunda usa etiquetas sintéticas para estimar una probabilidad y aprovecha la representación ya aprendida. Una regla de combinación, elegida antes de mirar el conjunto de prueba, produce la alerta final. La interfaz muestra ambos puntajes, la secuencia concreta, un mapa de atención y un texto explicativo. Ninguna salida sustituye una investigación ni una decisión legal.")
paragraph("El experimento usa una muestra determinista de remitentes del archivo IBM. La proporción positiva en prueba fue de {:.2%} por ventana. Sobre esa prueba retenida, el sistema combinado alcanzó precisión {:.3f}, exhaustividad {:.3f} y precisión promedio {:.4f}. Su valor debe leerse junto con el número de alertas ({}) y con la comparación de un clasificador idéntico que inició desde cero. Son resultados de un banco sintético, no tasas esperadas para Guatemala.".format(results["test_positive_rate"], results["test"]["combined"]["precision"], results["test"]["combined"]["recall"], results["test"]["combined"]["average_precision"], results["test"]["combined"]["alerts"]))

heading("2. Problema de negocio y marco guatemalteco")
paragraph("El enunciado plantea un corredor de remesas Guatemala–Estados Unidos de gran volumen. Para el área de cumplimiento, el problema práctico no es etiquetar una transferencia aislada como delito, sino ordenar una cola de señales, registrar la evidencia transaccional y decidir cuáles requieren investigación. Una regla estática de monto puede generar demasiadas alertas cuando el comportamiento legítimo varía entre clientes; también puede pasar por alto series de montos moderados o cambios bruscos de destino. Un modelo secuencial ofrece contexto, siempre que conserve trazabilidad y que su tasa de falsas alarmas sea compatible con la capacidad diaria del equipo.")
paragraph("En Guatemala, el Decreto 67-2001, Ley Contra el Lavado de Dinero u Otros Activos, y su reglamento establecen un marco para las personas obligadas. La Superintendencia de Bancos, mediante la Intendencia de Verificación Especial, cumple funciones preventivas y recibe reportes de transacciones sospechosas. El Decreto 58-2005 aborda la prevención y represión del financiamiento del terrorismo. La alerta de este prototipo es solo una señal interna para análisis; no equivale a un Reporte de Transacción Sospechosa ni determina culpabilidad. Un banco real necesitaría políticas aprobadas, diligencia sobre clientes, controles de acceso, conservación de evidencia y decisión humana documentada antes de escalar un caso.")
paragraph("El dataset IBM no describe migrantes, bancos guatemaltecos, agentes de remesas ni monedas del corredor. Por tanto, ninguna cifra de este reporte puede transferirse directamente a la obligación regulatoria o al volumen operativo local. La investigación de Altman y colaboradores (NeurIPS 2023) explica por qué los datos reales de lavado son difíciles de compartir y por qué las simulaciones permiten experimentar con etiquetas completas. Esa ventaja metodológica también impone una limitación: el simulador puede crear regularidades que no existan en clientes reales.")

heading("3. Datos, secuencias y decisiones de representación")
raw = results["raw_audit"]
wa = results["window_audit"]
paragraph("El archivo HI-Small contiene {:,} transacciones y {:,} transacciones etiquetadas como lavado. Seleccionamos el {} % de remitentes mediante un hash estable independiente de la etiqueta: la muestra produjo {:,} filas y {:,} remitentes. Este diseño evita escoger todos los casos positivos para inflar la prevalencia de evaluación. Un segundo hash, también por remitente, asigna aproximadamente 70 % a entrenamiento, 15 % a validación y 15 % a prueba. Ningún historial de un mismo remitente aparece en más de un grupo. El escalador se ajusta exclusivamente con entrenamiento, evitando que los valores de validación o prueba modifiquen la representación de entrada.".format(raw["raw_rows"], raw["raw_positive_transactions"], results["sample_percent"], raw["sample_rows"], raw["sample_senders"]))
paragraph("Cada remitente se ordena por marca de tiempo. Dividimos su historial en ventanas consecutivas no superpuestas de hasta 24 operaciones y descartamos ventanas unitarias, porque una sola operación no constituye patrón temporal. El proceso generó {:,} ventanas; {} fragmentos unitarios se excluyeron. La longitud limitada controla memoria y tiempo en Colab T4, aunque puede cortar una trayectoria ilícita que abarque más movimientos. La etiqueta de una ventana es positiva si al menos una de sus operaciones tiene etiqueta de lavado; esa regla sirve para priorización inicial y no identifica por sí misma cuál transacción fue decisiva.".format(wa["windows"], wa["dropped_singleton_windows"]))
paragraph("Las variables de cada movimiento son monto pagado en escala logarítmica, tiempo desde el movimiento anterior, hora codificada con seno y coseno, coincidencia o cambio de banco, diferencia de divisa, novedad del destino y tipo de pago codificado por categorías. La escala logarítmica reduce el dominio de montos extremos sin borrar diferencias pequeñas. La hora circular preserva cercanía entre 23:59 y 00:00. El intervalo y el destino nuevo capturan ritmo y dispersión que una observación individual omite. No entran la etiqueta, el ID del remitente, los saldos posteriores ni variables calculadas con transacciones futuras.")
paragraph("El conjunto indicado como principal en el enunciado, PaySim, resulta problemático para secuencias de remitente: sus identificadores de origen se repiten muy poco y su etiqueta representa fraude de dinero móvil, no lavado. Se eligió IBM como fuente principal del experimento temporal porque sí contiene historiales y etiquetas AML. Esta desviación del orden de datasets propuesto se deja explícita para que el evaluador pueda juzgarla. Una extensión razonable sería usar PaySim para un control transaccional separado, sin llamarlo detector secuencial ni combinar sin validación etiquetas de fraude y lavado.")

heading("4. Modelo, transferencia y selección de umbrales")
paragraph("La etapa A implementa un encoder GRU que lee secuencias de longitud variable mediante empaquetado, resume cada secuencia en un vector de 32 dimensiones y entrega ese vector a un decoder GRU que intenta reconstruir las variables originales. Se entrena exclusivamente con ventanas normales del grupo de entrenamiento. La pérdida es el error cuadrático medio por posición real; los ceros de relleno no aportan al gradiente ni al puntaje. Para comparar errores entre casos, el error de una secuencia se convierte en su percentil respecto a la distribución de errores de entrenamiento normal. El umbral que declara anomalía se selecciona en validación maximizando F2; no es un número escogido visualmente sobre la prueba.")
paragraph("La reconstrucción permite aprender una noción de normalidad cuando las etiquetas positivas son escasas, en línea con el uso de reconstrucción secuencial estudiado por Lai y colaboradores (NeurIPS 2023). No se supone que todo comportamiento raro sea ilícito. Una remesa poco frecuente, un cambio legítimo de destino o una operación empresarial atípica pueden elevar el error. Por esa razón, la etapa A aporta un score de revisión y no una acusación. También podría reconstruir demasiado bien un caso sospechoso parecido al tráfico normal, ocasionando falsos negativos; ese riesgo se mide en la ablación y se discute en los casos.")
paragraph("La etapa B copia los pesos del encoder previamente entrenado y ajusta todos los parámetros con etiquetas (fine tuning, estrategia de transferencia de la semana 9). Encima del GRU, una capa de atención asigna un peso a cada transacción válida, produce un resumen y alimenta un clasificador binario. Se usa entropía cruzada binaria con ponderación de la clase positiva igual a la raíz de la razón entre negativos y positivos, acotada a 40. El peso combate el desbalance sin magnificar de forma descontrolada unas pocas ventanas. Los logits se calibran mediante regresión logística en validación para que el número mostrado al analista tenga una interpretación probabilística experimental. La calibración depende de la prevalencia de este dataset y debe repetirse con datos reales.")
paragraph("Para probar el aporte de la transferencia se entrenó un clasificador de la misma arquitectura, misma semilla de control, mismas ventanas y mismas épocas, pero con encoder inicializado aleatoriamente. La comparación es una ablación: la diferencia principal es el aprendizaje previo de normalidad. En validación se evalúan pesos de mezcla para los scores A y B entre 0.05 y 0.5 y se fija el umbral final por F2. El peso elegido para la etapa A fue {:.2f}. La prueba se consulta solo después de cerrar esas decisiones. Se informa precisión promedio, que resume la curva precisión–exhaustividad en un problema raro, además de precisión, exhaustividad y F2 al umbral; la exactitud global habría sido engañosa ante una mayoría de normales.".format(results["selection"]["alpha"]))

heading("5. Resultados de la ablación en prueba retenida")
labels = [("Etapa A", "stage_a"), ("Etapa B transferida", "stage_b_transfer"),
          ("Control desde cero", "baseline_from_scratch"), ("Combinación A+B", "combined")]
data = [["Sistema", "AP", "Precisión", "Recall", "F2", "Alertas"]]
for label, key in labels:
    item = results["test"][key]
    data.append([label, f"{item['average_precision']:.4f}", f"{item['precision']:.3f}",
                 f"{item['recall']:.3f}", f"{item['f2']:.3f}", str(item["alerts"])])
    word_parts.append(" ".join(data[-1]))
table = Table(data, colWidths=[5.1 * cm, 2.0 * cm, 2.3 * cm, 2.0 * cm, 1.7 * cm, 2.0 * cm], repeatRows=1)
table.setStyle(TableStyle([
    ("BACKGROUND", (0, 0), (-1, 0), colors.white),
    ("TEXTCOLOR", (0, 0), (-1, 0), colors.black),
    ("FONTNAME", (0, 0), (-1, 0), "TimesLocalBold"),
    ("FONTNAME", (0, 1), (-1, -1), "TimesLocal"),
    ("FONTSIZE", (0, 0), (-1, -1), 9.5),
    ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ("TOPPADDING", (0, 0), (-1, -1), 6),
    ("LINEABOVE", (0, 0), (-1, 0), 0.5, colors.black),
    ("LINEBELOW", (0, 0), (-1, 0), 0.5, colors.black),
    ("LINEBELOW", (0, -1), (-1, -1), 0.5, colors.black),
]))
story.append(table)
story.append(Spacer(1, 8))
paragraph("AP es el área promedio de la curva precisión–exhaustividad; Recall es la proporción de ventanas positivas recuperadas. Los umbrales difieren por sistema y se eligieron con validación para F2. Por ello, la tabla informa tanto capacidad de ordenamiento (AP) como carga concreta de alertas. La diferencia entre el clasificador transferido y el control desde cero cuantifica el beneficio o la falta de beneficio del preentrenamiento; la diferencia entre etapa B y combinación muestra si el error de reconstrucción añade información útil después de clasificar. Una mejora marginal o negativa no se interpreta como éxito por decreto: puede revelar que la simulación permite aprender las etiquetas directamente o que el encoder de normalidad no conserva los rasgos que distinguen lavado.")
transfer_ap = results["test"]["stage_b_transfer"]["average_precision"]
baseline_ap = results["test"]["baseline_from_scratch"]["average_precision"]
combo_ap = results["test"]["combined"]["average_precision"]
paragraph("En este experimento, AP transferida = {:.4f}, AP desde cero = {:.4f} y AP combinada = {:.4f}. La diferencia transferida menos control es {:+.4f}; la diferencia combinada menos transferida es {:+.4f}. Esos números, y no la arquitectura propuesta por sí sola, determinan si las dos etapas aportaron valor en el subconjunto. No se repitieron semillas ni se estimaron intervalos de confianza, así que una diferencia pequeña no demuestra superioridad robusta. Para una decisión de inversión o producción haría falta repetir el experimento, probar datos de periodos posteriores y medir alertas por analista y por día.".format(transfer_ap, baseline_ap, combo_ap, transfer_ap - baseline_ap, combo_ap - transfer_ap))
paragraph("Al umbral de validación, la combinación obtuvo F2 {:.3f} frente a {:.3f} del control desde cero, con {} alertas frente a {}. Esa diferencia sugiere menor carga de revisión en esta prueba, aunque la AP del control fue ligeramente superior. La ganancia puntual no demuestra generalización a otros periodos ni a remesas reales.".format(results["test"]["combined"]["f2"], results["test"]["baseline_from_scratch"]["f2"], results["test"]["combined"]["alerts"], results["test"]["baseline_from_scratch"]["alerts"]))

heading("6. Cinco expedientes de prueba y trazabilidad")
tp = [c for c in cases if c["label"] and c["alert"]]
errors = [c for c in cases if c["label"] != int(c["alert"])]
selected = sorted(tp, key=lambda c: c["combined_score"], reverse=True)[:2]
if len(tp) >= 3:
    remaining = [c for c in tp if c not in selected]
    selected.append(min(remaining, key=lambda c: abs(c["combined_score"] - results["test"]["combined"]["threshold"])))
selected += sorted(errors, key=lambda c: abs(c["combined_score"] - results["test"]["combined"]["threshold"]))[:2]
for number, case in enumerate(selected[:5], 1):
    kind = "verdadero positivo" if case["label"] and case["alert"] else ("falso positivo" if case["alert"] else "falso negativo")
    attention = np.asarray(case["attention"])
    reconstruction = np.asarray(case["reconstruction_error"])
    a = int(attention.argmax())
    e = int(reconstruction.argmax())
    ta, te = case["transactions"][a], case["transactions"][e]
    amounts = np.asarray([t["amount"] for t in case["transactions"]])
    unique_dest = len({t["destination"] for t in case["transactions"]})
    labeled = sum(t["transaction_label"] for t in case["transactions"])
    label_text = "transacción marcada" if labeled == 1 else "transacciones marcadas"
    paragraph(f"<b>Caso {number}: {kind}; remitente {html.escape(case['sender'])}.</b> La ventana tiene {len(amounts)} movimientos, {unique_dest} destinos y {labeled} {label_text} como lavado por el generador. El mayor peso de atención recae en el movimiento {a+1} ({ta['amount']:,.2f} {html.escape(ta['currency'])}, {html.escape(ta['format'])}, {html.escape(ta['timestamp'])}); el mayor error de reconstrucción aparece en el movimiento {e+1} ({te['amount']:,.2f} {html.escape(te['currency'])}, {html.escape(te['format'])}). La mediana de montos de esta misma ventana es {np.median(amounts):,.2f}. Su score A es {case['anomaly_score']:.3f}, su probabilidad B es {case['stage_b_probability']:.2%} y la mezcla es {case['combined_score']:.3f}. {'La coincidencia de una etiqueta positiva con la alerta amerita revisar el recorrido de destinos y el ritmo de pagos; el mapa solo prioriza documentos, no prueba fraccionamiento ni intención.' if kind == 'verdadero positivo' else ('El modelo alertó pese a que la etiqueta sintética es normal. Un monto o destino infrecuente puede ser legítimo; este caso ilustra carga de revisión y la necesidad de contexto del cliente.' if kind == 'falso positivo' else 'La etiqueta sintética indica lavado y el modelo no superó el umbral. La señal pudo diluirse en operaciones normales de la ventana; hay que estudiar el episodio y la definición de ventana antes de cambiar el umbral.')}")
if len(selected) < 5:
    paragraph(f"La prueba produjo solo {len(selected)} expedientes que cumplen la combinación solicitada de tres verdaderos positivos y dos errores. No se fabricaron casos; esta escasez sería una limitación del experimento y exigiría ampliar el muestreo antes de cualquier conclusión operacional.")
paragraph("Los pesos de atención identifican posiciones que el clasificador usó para resumir su secuencia; el error de reconstrucción mide una diferencia respecto a patrones normales. Los dos mapas pueden discrepar y ninguno establece causalidad. Rigotti y colaboradores (ICLR 2022) muestran que interpretar atención requiere condiciones adicionales de fidelidad; nuestra arquitectura no impone esas condiciones. Por ello el párrafo automático del MVP dice qué transacciones revisar, describe puntuaciones y advierte de incertidumbre. Un analista debe contrastar contrapartes, documentación de origen de fondos y comportamiento histórico externo al dataset antes de escalar una sospecha.")

heading("7. Límites y ruta responsable hacia producción")
paragraph("Este estudio no valida una solución utilizable en una empresa de remesas. Primero, la población IBM combina transferencias y otros formatos bancarios, mientras el objetivo empresarial serían remesas transfronterizas. Segundo, una ventana positiva por OR puede incluir muchas transacciones normales y asignar una etiqueta amplia al remitente; la atención podría destacar una operación distinta de la etiquetada. Tercero, la división por hash de remitente impide filtración de identidad, pero todos los grupos proceden del mismo periodo sintético: no mide deriva temporal. Cuarto, la probabilidad se calibró en validación con una prevalencia específica; un cambio de mezcla de clientes o de producto alteraría su significado. Quinto, el score combinado se optimizó para F2, no para el costo real de investigaciones, capacidad disponible o severidad de pérdidas.")
paragraph("Una implementación real requeriría acuerdos de acceso y protección de datos, evaluación del flujo regulatorio con la IVE y asesores jurídicos, catálogo de señales autorizadas en el momento de decisión, trazabilidad de versiones, control de acceso y registro de cada alerta. El umbral debe fijarse con una meta de carga diaria y revisión de falsos negativos mediante muestreo humano; también conviene probar estabilidad por canal, región y perfil de cliente. Deben evaluarse cambios de comportamiento y posibles sesgos antes de desplegar. El benchmark de Liu y Paparrizos (NeurIPS 2024) advierte que fallos de datasets y métricas pueden crear una impresión falsa de progreso en detección de anomalías; por eso aquí reportamos la prevalencia, los splits, la ablación y los fallos junto a las métricas.")
paragraph("El MVP permite consultar remitentes retenidos y sus ventanas con salidas calculadas previamente. Así evita redistribuir el CSV masivo. Para operar en vivo harían falta ingestión segura, puntuación de nuevas transacciones, gestión de casos, reentrenamiento y validación prospectiva. El modelo debe apoyar a un experto capaz de desactivar alertas defectuosas y documentar su decisión.")

heading("8. Uso de IA generativa y decisiones del grupo")
paragraph("Se utilizó un asistente de IA para ayudar a redactar y programar el proyecto a partir del enunciado, con un prompt que pedía implementar los cuatro componentes, reproducibilidad, análisis de errores y presentación del MVP. Ese prompt funcionó al convertir criterios generales en archivos y verificaciones concretas. El grupo debe revisar antes de entregar las decisiones de dataset, longitud de ventana, variables, pesos de pérdida, umbrales, interpretación regulatoria y conclusiones. Las métricas proceden de una ejecución del código sobre IBM HI-Small, no de texto generado. La IA no sustituye la revisión de resultados ni la defensa oral de por qué se eligió cada técnica.")

heading("Referencias")
refs = [
    "Altman, E., Blanuša, J., von Niederhäusern, L., Egressy, B., Anghel, A., &amp; Atasu, K. (2023). Realistic Synthetic Financial Transactions for Anti-Money Laundering Models. <i>NeurIPS 36, Datasets and Benchmarks</i>. https://papers.neurips.cc/paper_files/paper/2023/hash/5f38404edff6f3f642d6fa5892479c42-Abstract-Datasets_and_Benchmarks.html",
    "Lai, C.-Y., Sun, F.-K., Gao, Z., Lang, J. H., &amp; Boning, D. S. (2023). Nominality Score Conditioned Time Series Anomaly Detection by Point/Sequential Reconstruction. <i>NeurIPS 36</i>. https://papers.neurips.cc/paper_files/paper/2023/hash/f1cf02ce09757f57c3b93c0db83181e0-Abstract-Conference.html",
    "Rigotti, M., Miksovic, C., Giurgiu, I., Gschwind, T., &amp; Scotton, P. (2022). Attention-based Interpretability with Concept Transformers. <i>ICLR 2022</i>. https://openreview.net/forum?id=kAa9eDS0RdO",
    "Liu, Q., &amp; Paparrizos, J. (2024). The Elephant in the Room: Towards A Reliable Time-Series Anomaly Detection Benchmark. <i>NeurIPS 37, Datasets and Benchmarks</i>. https://papers.neurips.cc/paper_files/paper/2024/file/c3f3c690b7a99fba16d0efd35cb83b2c-Paper-Datasets_and_Benchmarks_Track.pdf",
    "Congreso de la República de Guatemala. (2001). <i>Ley Contra el Lavado de Dinero u Otros Activos</i>, Decreto 67-2001. Superintendencia de Bancos. https://www.sib.gob.gt/c/document_library/get_file?folderId=7450432&amp;name=DLFE-36514.pdf",
    "Congreso de la República de Guatemala. (2005). <i>Ley para Prevenir y Reprimir el Financiamiento del Terrorismo</i>, Decreto 58-2005. Superintendencia de Bancos. https://www.sib.gob.gt/c/document_library/get_file?folderId=7450432&amp;name=DLFE-36522.pdf",
]
for ref in refs:
    paragraph(ref, "ReportSmall")

word_count = len(re.findall(r"\b[\wÁÉÍÓÚÜÑáéíóúüñ-]+\b", " ".join(word_parts)))
if not 2000 <= word_count <= 3000:
    raise RuntimeError(f"Executive report has {word_count} words; expected 2000–3000")


class ReportDoc(BaseDocTemplate):
    def __init__(self, filename):
        super().__init__(filename, pagesize=A4, leftMargin=2.54*cm, rightMargin=2.54*cm,
                         topMargin=2.54*cm, bottomMargin=2.54*cm)
        frame = Frame(self.leftMargin, self.bottomMargin, self.width, self.height,
                      leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0)
        self.addPageTemplates([PageTemplate(id="cover", frames=frame, onPage=self.cover),
                               PageTemplate(id="body", frames=frame, onPage=self.decor)])

    def cover(self, canvas, doc):
        canvas.saveState()
        center = A4[0] / 2
        canvas.setFillColor(colors.black)
        canvas.setFont("TimesLocalBold", 12)
        canvas.drawCentredString(center, A4[1] - 2.6*cm,
                                 "UNIVERSIDAD DEL VALLE DE GUATEMALA")
        canvas.setFont("TimesLocal", 12)
        canvas.drawCentredString(center, A4[1] - 3.25*cm, "CC3092 - Deep Learning")
        logo = ROOT / "assets/uvg_logo.png"
        canvas.drawImage(str(logo), center - 2.45*cm, A4[1] - 13.2*cm,
                         width=4.9*cm, height=7.25*cm, mask="auto")
        canvas.setFont("TimesLocalBold", 15)
        canvas.drawCentredString(center, A4[1] - 15.75*cm,
                                 "Informe ejecutivo Proyecto 2")
        canvas.setFont("TimesLocal", 12)
        canvas.drawCentredString(center, A4[1] - 19.75*cm, "Diego Patzan - 23525")
        canvas.drawCentredString(center, A4[1] - 20.4*cm, "Ihan Marroquin - 23108")
        canvas.setFont("TimesLocalBold", 12)
        canvas.drawCentredString(center, A4[1] - 24.6*cm,
                                 "GUATEMALA, 20 de septiembre de 2026")
        canvas.restoreState()

    def decor(self, canvas, doc):
        canvas.saveState()
        canvas.setFont("TimesLocal", 9)
        canvas.setFillColor(colors.black)
        canvas.drawCentredString(A4[0]/2, 1.5*cm, str(doc.page))
        canvas.restoreState()


ReportDoc(str(target)).build(story)
print(target)
print("words", word_count)
