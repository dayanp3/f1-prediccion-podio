"""Fase 5 - Pipeline de machine learning e inferencia.

Entrena una regresion logistica que predice si un piloto termina en el
podio, usando data/pilotos_analitica.csv.

Pasos:
  1. Auditoria de nulos e imputacion con la mediana (calculada solo en train).
  2. Separacion de X e y con split cronologico: train 2014-2023, test 2024.
  3. Escalado con StandardScaler ajustado solo en train.
  4. Regresion logistica (solver lbfgs).
  5. Evaluacion: reporte de clasificacion, matriz de confusion, ROC-AUC y
     comparacion contra un baseline.
  6. Funcion de inferencia con umbral 0.60 para decidir si se apuesta.

Uso (desde la carpeta F1):
    python src/04_modelado_podio.py
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

RAIZ: Path = Path(__file__).resolve().parent.parent
RUTA_CSV: Path = RAIZ / "data" / "pilotos_analitica.csv"
RUTA_MODELO: Path = RAIZ / "docs" / "modelo_podio.json"

VARIABLES: list[str] = [
    "posicion_salida",
    "puntos_piloto_previos",
    "puntos_escuderia_previos",
    "podios_ultimas_5",
    "pos_promedio_ultimas_5",
]
OBJETIVO: str = "podio"
TEMPORADA_TEST: int = 2024
UMBRAL: float = 0.60


@dataclass(frozen=True)
class Prediccion:
    """Resultado de la funcion de inferencia para un piloto en una carrera."""

    probabilidad: float
    umbral: float
    apostar: bool

    @property
    def decision(self) -> str:
        return "APOSTAR al podio" if self.apostar else "NO apostar"


def titulo(texto: str) -> None:
    print()
    print("=" * 70)
    print(texto)
    print("=" * 70)


def cargar_y_auditar(ruta_csv: Path) -> pd.DataFrame:
    """Paso 1a: lee la matriz analitica y reporta los nulos."""
    tabla = pd.read_csv(ruta_csv, encoding="utf-8-sig")
    titulo("1. AUDITORIA DE NULOS")
    print(f"Filas: {len(tabla):,} | columnas: {tabla.shape[1]}")
    nulos = tabla[VARIABLES].isna().sum()
    for columna, cantidad in nulos.items():
        print(f"  {columna:<26}{cantidad:>4} nulos ({cantidad / len(tabla):.2%})")
    return tabla


def separar(tabla: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """Paso 2: X e y con split cronologico (se entrena con el pasado)."""
    train = tabla[tabla["temporada"] < TEMPORADA_TEST]
    test = tabla[tabla["temporada"] == TEMPORADA_TEST]
    titulo("2. SPLIT CRONOLOGICO")
    print(f"  Train: temporadas {train['temporada'].min()} a {train['temporada'].max()} "
          f"| {len(train):,} filas | {train[OBJETIVO].mean():.2%} podios")
    print(f"  Test:  temporada {TEMPORADA_TEST} | {len(test):,} filas | {test[OBJETIVO].mean():.2%} podios")
    return train[VARIABLES], train[OBJETIVO], test[VARIABLES], test[OBJETIVO]


def entrenar(x_train: pd.DataFrame, y_train: pd.Series) -> Pipeline:
    """Pasos 1b, 3 y 4: imputacion, escalado y regresion logistica.

    Al estar dentro de un Pipeline, la mediana y la escala se calculan
    unicamente con los datos de entrenamiento.
    """
    modelo = Pipeline(
        steps=[
            ("imputacion", SimpleImputer(strategy="median")),
            ("escalado", StandardScaler()),
            ("regresion", LogisticRegression(solver="lbfgs", max_iter=1000)),
        ]
    )
    modelo.fit(x_train, y_train)

    titulo("3. IMPUTACION, ESCALADO Y REGRESION LOGISTICA (ajustados solo en train)")
    medianas = modelo.named_steps["imputacion"].statistics_
    escala = modelo.named_steps["escalado"]
    regresion = modelo.named_steps["regresion"]
    print(f"  {'Variable':<26}{'Mediana':>9}{'Media':>9}{'Desv.':>9}{'Coeficiente':>13}")
    for i, variable in enumerate(VARIABLES):
        print(f"  {variable:<26}{medianas[i]:>9.2f}{escala.mean_[i]:>9.2f}"
              f"{escala.scale_[i]:>9.2f}{regresion.coef_[0][i]:>13.3f}")
    print(f"  Intercepto: {regresion.intercept_[0]:.3f}")
    print("  Coeficiente positivo = sube la probabilidad de podio; negativo = la baja.")
    return modelo


def resumen_metricas(y_real: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "exactitud": accuracy_score(y_real, y_pred),
        "precision": precision_score(y_real, y_pred, zero_division=0),
        "recall": recall_score(y_real, y_pred, zero_division=0),
        "f1": f1_score(y_real, y_pred, zero_division=0),
    }


def evaluar(modelo: Pipeline, x_test: pd.DataFrame, y_test: pd.Series) -> dict:
    """Paso 5: metricas del modelo y comparacion contra el baseline."""
    probabilidades = modelo.predict_proba(x_test)[:, 1]
    pred_050 = (probabilidades >= 0.5).astype(int)
    pred_umbral = (probabilidades >= UMBRAL).astype(int)

    titulo(f"4. EVALUACION EN TEST (temporada {TEMPORADA_TEST})")
    print("Reporte de clasificacion (umbral estandar 0.50):")
    print(classification_report(y_test, pred_050, target_names=["No podio", "Podio"], digits=3))

    matriz = confusion_matrix(y_test, pred_050)
    print("Matriz de confusion (umbral 0.50):")
    print(f"  {'':<16}{'Predijo NO':>12}{'Predijo SI':>12}")
    print(f"  {'Real NO podio':<16}{matriz[0][0]:>12}{matriz[0][1]:>12}")
    print(f"  {'Real SI podio':<16}{matriz[1][0]:>12}{matriz[1][1]:>12}")

    auc_modelo = roc_auc_score(y_test, probabilidades)
    print(f"\nROC-AUC del modelo: {auc_modelo:.4f}")

    # Baseline: "hacen podio los 3 primeros de la parrilla".
    # Las salidas nulas (pit lane) se tratan como ultimo puesto.
    salida = x_test["posicion_salida"].fillna(99)
    pred_base = (salida <= 3).astype(int)
    auc_base = roc_auc_score(y_test, -salida)
    pred_nadie = np.zeros(len(y_test), dtype=int)

    titulo("5. COMPARACION CONTRA EL BASELINE")
    filas = {
        "Nadie hace podio": resumen_metricas(y_test, pred_nadie),
        "Baseline: salen top 3": resumen_metricas(y_test, pred_base),
        "Modelo (umbral 0.50)": resumen_metricas(y_test, pred_050),
        f"Modelo (umbral {UMBRAL:.2f})": resumen_metricas(y_test, pred_umbral),
    }
    print(f"  {'Estrategia':<24}{'Exactitud':>10}{'Precision':>10}{'Recall':>10}{'F1':>10}")
    for nombre, m in filas.items():
        print(f"  {nombre:<24}{m['exactitud']:>10.3f}{m['precision']:>10.3f}{m['recall']:>10.3f}{m['f1']:>10.3f}")
    print(f"\n  ROC-AUC baseline (solo posicion de salida): {auc_base:.4f}")
    print(f"  ROC-AUC modelo (5 variables):               {auc_modelo:.4f}")

    apuestas = int(pred_umbral.sum())
    aciertos = int(((pred_umbral == 1) & (y_test == 1)).sum())
    print(f"\n  Con el umbral {UMBRAL:.2f} el modelo habria apostado {apuestas} veces en {TEMPORADA_TEST} "
          f"y acertado {aciertos} ({aciertos / apuestas:.1%}).")

    return {
        "matriz_confusion_050": matriz.tolist(),
        "roc_auc_modelo": auc_modelo,
        "roc_auc_baseline": auc_base,
        "estrategias": filas,
        "apuestas_umbral": apuestas,
        "aciertos_umbral": aciertos,
    }


def predecir_podio(
    modelo: Pipeline,
    posicion_salida: float | None,
    puntos_piloto_previos: float,
    puntos_escuderia_previos: float,
    podios_ultimas_5: int,
    pos_promedio_ultimas_5: float | None,
    umbral: float = UMBRAL,
) -> Prediccion:
    """Paso 6 - Funcion de inferencia.

    Recibe los datos de un piloto antes de la carrera y devuelve la
    probabilidad de podio y la decision. Acepta None en los datos que
    pueden faltar: el pipeline los imputa con la mediana.
    """
    caso = pd.DataFrame(
        [[posicion_salida, puntos_piloto_previos, puntos_escuderia_previos,
          podios_ultimas_5, pos_promedio_ultimas_5]],
        columns=VARIABLES,
        dtype="float64",
    )
    probabilidad = float(modelo.predict_proba(caso)[0, 1])
    return Prediccion(probabilidad=probabilidad, umbral=umbral, apostar=probabilidad >= umbral)


def demostrar_inferencia(modelo: Pipeline) -> list[dict]:
    """Prueba la funcion de inferencia con tres casos tipo."""
    titulo(f"6. FUNCION DE INFERENCIA (umbral {UMBRAL:.2f})")
    casos = [
        ("A. Favorito claro: sale 1, lider del campeonato, 4 podios seguidos",
         dict(posicion_salida=1, puntos_piloto_previos=300, puntos_escuderia_previos=500,
              podios_ultimas_5=4, pos_promedio_ultimas_5=2.0)),
        ("B. Caso parejo: sale 4, buen equipo, 2 podios recientes",
         dict(posicion_salida=4, puntos_piloto_previos=180, puntos_escuderia_previos=350,
              podios_ultimas_5=2, pos_promedio_ultimas_5=5.0)),
        ("C. Sin opciones: sale 15, equipo de mitad de tabla, sin podios",
         dict(posicion_salida=15, puntos_piloto_previos=12, puntos_escuderia_previos=30,
              podios_ultimas_5=0, pos_promedio_ultimas_5=13.0)),
    ]
    salida: list[dict] = []
    for nombre, datos in casos:
        resultado = predecir_podio(modelo, **datos)
        print(f"  {nombre}")
        print(f"     Probabilidad de podio: {resultado.probabilidad:.1%} -> {resultado.decision}")
        salida.append({"caso": nombre, "datos": datos,
                       "probabilidad": resultado.probabilidad, "apostar": resultado.apostar})
    return salida


def guardar_modelo(modelo: Pipeline, metricas: dict, casos: list[dict]) -> None:
    """Guarda los parametros del modelo para la pagina web (simulador)."""
    contenido = {
        "variables": VARIABLES,
        "umbral": UMBRAL,
        "medianas": modelo.named_steps["imputacion"].statistics_.tolist(),
        "medias": modelo.named_steps["escalado"].mean_.tolist(),
        "desviaciones": modelo.named_steps["escalado"].scale_.tolist(),
        "coeficientes": modelo.named_steps["regresion"].coef_[0].tolist(),
        "intercepto": float(modelo.named_steps["regresion"].intercept_[0]),
        "metricas": metricas,
        "casos": casos,
    }
    RUTA_MODELO.write_text(json.dumps(contenido, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nParametros del modelo guardados en {RUTA_MODELO.relative_to(RAIZ)}")


def main() -> None:
    tabla = cargar_y_auditar(RUTA_CSV)
    x_train, y_train, x_test, y_test = separar(tabla)
    modelo = entrenar(x_train, y_train)
    metricas = evaluar(modelo, x_test, y_test)
    casos = demostrar_inferencia(modelo)
    guardar_modelo(modelo, metricas, casos)


if __name__ == "__main__":
    main()
