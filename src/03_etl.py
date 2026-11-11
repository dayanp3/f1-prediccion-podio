"""Fase 3 - ETL: del almacen relacional a la matriz analitica plana.

Extrae de data/f1.sqlite las temporadas 2014 a 2024, cruza las tablas para
traer nombres legibles, calcula variables que se conocen ANTES de la carrera
y exporta un solo archivo plano: data/pilotos_analitica.csv

Cada fila es un piloto en una carrera. La columna objetivo es `podio`.

Uso (desde la carpeta F1):
    python src/03_etl.py
"""

from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path

import pandas as pd

RAIZ: Path = Path(__file__).resolve().parent.parent
RUTA_DB: Path = RAIZ / "data" / "f1.sqlite"
RUTA_CSV: Path = RAIZ / "data" / "pilotos_analitica.csv"

ANIO_INICIO: int = 2014
ANIO_FIN: int = 2024

# EXTRACCION: una sola consulta con los JOIN y las variables calculadas.
CONSULTA: str = """
WITH historial AS (
    -- Forma reciente: se calcula sobre TODA la historia del piloto para que
    -- las primeras carreras de 2014 ya tengan datos de 2013.
    -- La ventana termina en la carrera anterior (1 PRECEDING): nunca
    -- incluye la carrera que se quiere predecir.
    SELECT r.resultId,
           SUM(r.positionOrder <= 3) OVER ultimas_5 AS podios_ultimas_5,
           AVG(r.positionOrder)      OVER ultimas_5 AS pos_promedio_ultimas_5
    FROM results r
    JOIN races ra ON ra.raceId = r.raceId
    WINDOW ultimas_5 AS (
        PARTITION BY r.driverId
        ORDER BY ra.date
        ROWS BETWEEN 5 PRECEDING AND 1 PRECEDING
    )
)
SELECT r.resultId                         AS resultado_id,
       ra.year                            AS temporada,
       ra.round                           AS ronda,
       ra.date                            AS fecha,
       ra.name                            AS gran_premio,
       d.forename || ' ' || d.surname     AS piloto,
       c.name                             AS escuderia,
       r.grid                             AS posicion_salida,
       COALESCE(ds.points, 0)             AS puntos_piloto_previos,
       COALESCE(cs.points, 0)             AS puntos_escuderia_previos,
       COALESCE(h.podios_ultimas_5, 0)    AS podios_ultimas_5,
       h.pos_promedio_ultimas_5           AS pos_promedio_ultimas_5,
       r.positionOrder                    AS posicion_final
FROM results r
JOIN races ra        ON ra.raceId = r.raceId
JOIN drivers d       ON d.driverId = r.driverId
JOIN constructors c  ON c.constructorId = r.constructorId
JOIN historial h     ON h.resultId = r.resultId
-- Puntos del campeonato tal como estaban al terminar la carrera ANTERIOR
LEFT JOIN races anterior
       ON anterior.year = ra.year AND anterior.round = ra.round - 1
LEFT JOIN driver_standings ds
       ON ds.raceId = anterior.raceId AND ds.driverId = r.driverId
LEFT JOIN constructor_standings cs
       ON cs.raceId = anterior.raceId AND cs.constructorId = r.constructorId
WHERE ra.year BETWEEN ? AND ?
ORDER BY ra.date, r.positionOrder
"""


def extraer(ruta_db: Path) -> pd.DataFrame:
    """Lee del almacen (solo lectura) las filas de la era con sus JOIN."""
    uri = f"{ruta_db.resolve().as_uri()}?mode=ro"
    with sqlite3.connect(uri, uri=True) as conexion:
        return pd.read_sql(CONSULTA, conexion, params=(ANIO_INICIO, ANIO_FIN))


def transformar(tabla: pd.DataFrame) -> pd.DataFrame:
    """Limpia anomalias y crea la variable objetivo."""
    tabla = tabla.copy()

    # Anomalia detectada en la auditoria: salida 0 = pit lane, no es una
    # posicion real. Se deja como nulo para que el pipeline la impute.
    tabla["posicion_salida"] = tabla["posicion_salida"].astype("Int64")
    tabla.loc[tabla["posicion_salida"] == 0, "posicion_salida"] = pd.NA

    tabla["pos_promedio_ultimas_5"] = tabla["pos_promedio_ultimas_5"].round(2)

    # Variable objetivo (dependiente): 1 si termino entre los 3 primeros.
    tabla["podio"] = (tabla["posicion_final"] <= 3).astype(int)
    return tabla


def validar(tabla: pd.DataFrame, ruta_db: Path) -> None:
    """Comprueba que el ETL no perdio ni duplico filas."""
    uri = f"{ruta_db.resolve().as_uri()}?mode=ro"
    with sqlite3.connect(uri, uri=True) as conexion:
        esperadas = conexion.execute(
            "SELECT COUNT(*) FROM results r JOIN races ra USING (raceId) "
            "WHERE ra.year BETWEEN ? AND ?",
            (ANIO_INICIO, ANIO_FIN),
        ).fetchone()[0]

    podios_por_carrera = tabla.groupby(["temporada", "ronda"])["podio"].sum()
    print("VALIDACIONES")
    print(f"  Filas en el almacen para la era: {esperadas:,} | filas en la matriz: {len(tabla):,}")
    print(f"  resultado_id repetidos: {tabla['resultado_id'].duplicated().sum()}")
    print(f"  Carreras con exactamente 3 podios: {(podios_por_carrera == 3).sum()} de {len(podios_por_carrera)}")
    assert len(tabla) == esperadas, "Los JOIN perdieron o duplicaron filas"
    assert tabla["resultado_id"].is_unique, "Hay resultados repetidos"
    assert (podios_por_carrera == 3).all(), "Hay carreras sin 3 podios"


def resumir(tabla: pd.DataFrame, ruta_csv: Path) -> None:
    """Imprime el resumen de la matriz analitica."""
    print()
    print("MATRIZ ANALITICA - pilotos_analitica.csv")
    print(f"  {len(tabla):,} filas x {tabla.shape[1]} columnas | {ruta_csv.stat().st_size / 1000:.0f} KB")
    print(f"  Temporadas {tabla['temporada'].min()} a {tabla['temporada'].max()}, "
          f"{tabla.groupby(['temporada', 'ronda']).ngroups} carreras, "
          f"{tabla['piloto'].nunique()} pilotos, {tabla['escuderia'].nunique()} escuderias")

    print()
    print("NULOS POR COLUMNA (solo las que tienen)")
    nulos = tabla.isna().sum()
    for columna, cantidad in nulos[nulos > 0].items():
        print(f"  {columna:<26}{cantidad:>5}  ({cantidad / len(tabla):.2%})")

    print()
    print("VARIABLE OBJETIVO podio")
    print(f"  1 (podio):    {tabla['podio'].sum():>5,}  ({tabla['podio'].mean():.2%})")
    print(f"  0 (no podio): {(1 - tabla['podio']).sum():>5,}  ({1 - tabla['podio'].mean():.2%})")

    print()
    print("EJEMPLO: los 5 primeros de la ultima carrera")
    ultima = tabla[tabla["fecha"] == tabla["fecha"].max()].head(5)
    columnas = ["gran_premio", "piloto", "posicion_salida", "puntos_piloto_previos",
                "podios_ultimas_5", "pos_promedio_ultimas_5", "posicion_final", "podio"]
    print(ultima[columnas].to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(description="ETL de f1.sqlite a pilotos_analitica.csv")
    parser.add_argument("--db", type=Path, default=RUTA_DB, help="Ruta del archivo SQLite")
    argumentos = parser.parse_args()

    tabla = transformar(extraer(argumentos.db))
    validar(tabla, argumentos.db)

    # CARGA: archivo plano. utf-8-sig para que Excel muestre bien las tildes.
    # lineterminator fijo para que el archivo pese igual en Windows y Linux.
    tabla.to_csv(RUTA_CSV, index=False, encoding="utf-8-sig", lineterminator="\n")
    resumir(tabla, RUTA_CSV)
    print()
    print(f"Archivo guardado en {RUTA_CSV.relative_to(RAIZ)}")


if __name__ == "__main__":
    main()
