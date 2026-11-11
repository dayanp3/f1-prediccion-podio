# Predicción de podio en Fórmula 1

Actividad de Inteligencia de Negocios (tercer corte): del almacén transaccional a un modelo predictivo y a una decisión.

**Pregunta del modelo:** antes de que empiece una carrera, ¿terminará este piloto en el podio?
**Decisión:** si la probabilidad es de 60 % o más, se apuesta al podio del piloto.

## Flujo del proyecto

| Etapa | Qué se hace | Script | Resultado |
|---|---|---|---|
| 1. Almacén transaccional | Carga los 14 CSV de Kaggle a SQLite | `src/01_cargar_sqlite.py` | `data/f1.sqlite` (14 tablas, 701.433 filas) |
| 2. Auditoría | Revisa llaves, nulos y anomalías | `src/02_auditoria.py` | `docs/auditoria.txt` |
| 3. ETL | Filtra 2014-2024, cruza tablas y calcula variables | `src/03_etl.py` | `data/pilotos_analitica.csv` (4.626 filas x 14 columnas) |
| 4. Pipeline ML | Imputación, split cronológico, escalado y regresión logística | `src/04_modelado_podio.py` | Métricas en consola y `docs/modelo_podio.json` |
| 5. Explotación | Función de inferencia con umbral 0,60 | `src/04_modelado_podio.py` | Decisión: apostar o no apostar |

## Datos

Dataset **Formula 1 World Championship (1950 - 2024)** de Rohan Rao, en Kaggle:
https://www.kaggle.com/datasets/rohanrao/formula-1-world-championship-1950-2020

Los CSV originales y el archivo `f1.sqlite` no se suben al repositorio porque se pueden regenerar. La matriz analítica `data/pilotos_analitica.csv` sí está incluida, así que el modelo se puede ejecutar sin descargar nada.

## Variables

**Independientes (se conocen antes de la carrera):**

| Variable | Qué significa |
|---|---|
| `posicion_salida` | Puesto en la parrilla de salida |
| `puntos_piloto_previos` | Puntos del piloto en el campeonato antes de la carrera |
| `puntos_escuderia_previos` | Puntos de la escudería antes de la carrera |
| `podios_ultimas_5` | Podios del piloto en sus 5 carreras anteriores |
| `pos_promedio_ultimas_5` | Puesto final promedio en esas 5 carreras |

**Dependiente:** `podio` (1 si terminó entre los 3 primeros, 0 si no).

El detalle está en `docs/variables.xlsx`.

## Cómo ejecutarlo

Requiere Python 3.10 o superior.

```
pip install -r requirements.txt
```

Para ejecutar solo el modelo (usa la matriz incluida):

```
python src/04_modelado_podio.py
```

Para reproducir todo desde cero, descarga el dataset de Kaggle, descomprime los CSV en `data/raw/` y ejecuta en orden:

```
python src/01_cargar_sqlite.py
python src/02_auditoria.py
python src/03_etl.py
python src/04_modelado_podio.py
```

## Resultados

Entrenamiento con las temporadas 2014 a 2023 (4.147 filas) y prueba con la temporada 2024 (479 filas, 72 podios).

| Estrategia | Exactitud | Precisión | Recall | F1 |
|---|---|---|---|---|
| Nadie hace podio | 0,850 | 0 | 0 | 0 |
| Baseline: los 3 primeros de la parrilla | 0,900 | 0,667 | 0,667 | 0,667 |
| Modelo, umbral 0,50 | 0,891 | 0,647 | 0,611 | 0,629 |
| Modelo, umbral 0,60 | 0,889 | 0,667 | 0,528 | 0,589 |

- ROC-AUC del modelo: 0,925. ROC-AUC del baseline: 0,921.
- Con el umbral 0,60 el modelo habría apostado 57 veces en 2024 y acertado 38 (66,7 %).

**Conclusión:** el modelo queda prácticamente empatado con la regla simple de la parrilla. La posición de salida pesa cuatro veces más que cualquier otra variable. Lo que aporta el modelo es una probabilidad para cualquier piloto, que es lo que necesita la función de inferencia para decidir.

## Limitaciones

- Solo el 14,79 % de las filas son podio, así que el porcentaje de acierto por sí solo engaña. Por eso se compara con ROC-AUC y contra un baseline.
- Las salidas desde el pit lane (posición 0 en el dataset) se tratan como nulo y se imputan con la mediana, que las ubica a mitad de parrilla aunque en realidad arrancan últimas.
- El modelo no usa clima, estrategia de neumáticos ni tiempos de clasificación.

## Página interactiva

`web/index.html` se abre con doble clic. Tiene cuatro secciones: el recorrido del flujo paso a paso, las variables, un simulador de la función de inferencia y un auto de F1 explicado por partes.

## Estructura

```
F1/
  data/
    pilotos_analitica.csv   Matriz analítica plana
    raw/                    CSV de Kaggle (no incluidos)
  docs/
    auditoria.txt           Reporte de la auditoría
    modelo_podio.json       Parámetros y métricas del modelo
    variables.xlsx          Variables independientes y dependiente
  src/                      Los 4 scripts del flujo
  web/index.html            Página interactiva
  requirements.txt
```

## Herramientas

Python, SQLite, pandas, scikit-learn y openpyxl. Desarrollado con apoyo de Claude como agente de IA.

## Autor

Dayan Gómez - Ingeniería de Sistemas, Institución Universitaria del Putumayo.
