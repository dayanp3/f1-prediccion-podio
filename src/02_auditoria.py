"""Fase 2 - Auditoria del almacen transaccional.

Revisa la calidad de los datos de data/f1.sqlite antes de hacer el ETL:
integridad de llaves, valores nulos, anomalias y cobertura por temporada.
Con eso se justifica que temporadas se usan para el modelo.

El reporte se imprime en pantalla y se guarda en docs/auditoria.txt.

Uso (desde la carpeta F1):
    python src/02_auditoria.py
"""

from __future__ import annotations

import argparse
import io
import sqlite3
from contextlib import redirect_stdout
from pathlib import Path

import pandas as pd

RAIZ: Path = Path(__file__).resolve().parent.parent
RUTA_DB: Path = RAIZ / "data" / "f1.sqlite"
RUTA_REPORTE: Path = RAIZ / "docs" / "auditoria.txt"

# Rango de temporadas propuesto para el modelo (era hibrida de la F1)
ANIO_INICIO: int = 2014
ANIO_FIN: int = 2024
ERA: str = f"ra.year BETWEEN {ANIO_INICIO} AND {ANIO_FIN}"


def conectar_solo_lectura(ruta_db: Path) -> sqlite3.Connection:
    """Abre la base en modo solo lectura: la auditoria no modifica nada."""
    return sqlite3.connect(f"{ruta_db.resolve().as_uri()}?mode=ro", uri=True)


def titulo(texto: str) -> None:
    print()
    print("=" * 70)
    print(texto)
    print("=" * 70)


def mostrar(conexion: sqlite3.Connection, sql: str) -> pd.DataFrame:
    """Ejecuta una consulta, imprime el resultado como tabla y lo devuelve."""
    tabla = pd.read_sql(sql, conexion)
    print(tabla.to_string(index=False))
    return tabla


def auditar_integridad(conexion: sqlite3.Connection) -> None:
    titulo("1. INTEGRIDAD: las llaves de results existen en los catalogos?")
    mostrar(
        conexion,
        """
        SELECT 'results.raceId -> races' AS relacion,
               SUM(raceId NOT IN (SELECT raceId FROM races)) AS huerfanos FROM results
        UNION ALL
        SELECT 'results.driverId -> drivers',
               SUM(driverId NOT IN (SELECT driverId FROM drivers)) FROM results
        UNION ALL
        SELECT 'results.constructorId -> constructors',
               SUM(constructorId NOT IN (SELECT constructorId FROM constructors)) FROM results
        """,
    )
    print("0 huerfanos = todos los JOIN van a cruzar sin perder filas.")


def auditar_nulos(conexion: sqlite3.Connection) -> None:
    titulo("2. NULOS por columna en results (tabla central)")
    resultados = pd.read_sql("SELECT * FROM results", conexion)
    nulos = resultados.isna().sum()
    reporte = pd.DataFrame(
        {
            "columna": nulos.index,
            "nulos": nulos.values,
            "porcentaje": (nulos.values / len(resultados) * 100).round(2),
        }
    )
    print(reporte.to_string(index=False))
    print(f"Total de filas: {len(resultados):,}")
    print("position nula = el piloto no termino la carrera (abandono, descalificado).")


def auditar_anomalias(conexion: sqlite3.Connection) -> None:
    titulo("3. ANOMALIAS por decada")
    mostrar(
        conexion,
        """
        WITH repetidos AS (
            SELECT raceId, driverId FROM results GROUP BY raceId, driverId HAVING COUNT(*) > 1
        ),
        podios AS (
            SELECT raceId, SUM(positionOrder <= 3) AS n_podios FROM results GROUP BY raceId
        )
        SELECT (ra.year / 10) * 10 AS decada,
               COUNT(DISTINCT ra.raceId) AS carreras,
               COUNT(*) AS filas,
               SUM(r.grid = 0) AS salida_en_0,
               (SELECT COUNT(*) FROM repetidos x JOIN races y USING (raceId)
                 WHERE (y.year / 10) * 10 = (ra.year / 10) * 10) AS piloto_repetido,
               (SELECT COUNT(*) FROM podios x JOIN races y USING (raceId)
                 WHERE x.n_podios <> 3 AND (y.year / 10) * 10 = (ra.year / 10) * 10) AS carreras_sin_3_podios
        FROM results r JOIN races ra USING (raceId)
        GROUP BY decada ORDER BY decada
        """,
    )
    print("salida_en_0: grid = 0 no es una posicion real (el piloto salio desde el pit lane).")
    print("piloto_repetido: entre 1950 y 1978 un piloto podia aparecer dos veces en una carrera.")
    print("carreras_sin_3_podios: autos compartidos en los 50, mas de 3 pilotos en el podio.")


def auditar_cobertura(conexion: sqlite3.Connection) -> None:
    titulo("4. COBERTURA por temporada (2010 en adelante)")
    mostrar(
        conexion,
        """
        SELECT ra.year AS temporada,
               COUNT(DISTINCT ra.raceId) AS carreras,
               COUNT(*) AS filas,
               ROUND(1.0 * COUNT(*) / COUNT(DISTINCT ra.raceId), 1) AS pilotos_por_carrera,
               SUM(r.grid = 0) AS salida_en_0,
               SUM(r.positionOrder <= 3) AS podios,
               MAX(r.points) AS max_puntos
        FROM results r JOIN races ra USING (raceId)
        WHERE ra.year >= 2010
        GROUP BY ra.year ORDER BY ra.year
        """,
    )
    print("max_puntos = 50 en 2014: la ultima carrera de esa temporada dio puntos dobles.")


def auditar_era_elegida(conexion: sqlite3.Connection) -> None:
    titulo(f"5. DECISION: temporadas {ANIO_INICIO} a {ANIO_FIN} (era hibrida)")
    resumen = mostrar(
        conexion,
        f"""
        SELECT COUNT(*) AS filas,
               COUNT(DISTINCT r.raceId) AS carreras,
               COUNT(DISTINCT r.driverId) AS pilotos,
               COUNT(DISTINCT r.constructorId) AS escuderias,
               SUM(r.grid = 0) AS salida_en_0,
               SUM(r.positionOrder <= 3) AS podios
        FROM results r JOIN races ra USING (raceId)
        WHERE {ERA}
        """,
    ).iloc[0]
    print(f"Salidas en 0 a tratar como nulo: {resumen.salida_en_0 / resumen.filas:.2%} de las filas.")
    print(f"Filas con podio (clase positiva): {resumen.podios / resumen.filas:.2%}.")

    print()
    print("Faltan datos de campeonato para alguna fila de la era?")
    mostrar(
        conexion,
        f"""
        SELECT SUM(NOT EXISTS (SELECT 1 FROM driver_standings d
                                WHERE d.raceId = r.raceId AND d.driverId = r.driverId)) AS sin_puntos_piloto,
               SUM(NOT EXISTS (SELECT 1 FROM constructor_standings c
                                WHERE c.raceId = r.raceId AND c.constructorId = r.constructorId)) AS sin_puntos_escuderia
        FROM results r JOIN races ra USING (raceId)
        WHERE {ERA}
        """,
    )

    print()
    print("Tasa de podio segun la posicion de salida (primeras 8 y pit lane):")
    mostrar(
        conexion,
        f"""
        SELECT r.grid AS salida, COUNT(*) AS veces, ROUND(AVG(r.positionOrder <= 3) * 100, 1) AS pct_podio
        FROM results r JOIN races ra USING (raceId)
        WHERE {ERA} AND r.grid <= 8
        GROUP BY r.grid ORDER BY r.grid
        """,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Auditoria del almacen f1.sqlite.")
    parser.add_argument("--db", type=Path, default=RUTA_DB, help="Ruta del archivo SQLite")
    argumentos = parser.parse_args()

    salida = io.StringIO()
    with redirect_stdout(salida), conectar_solo_lectura(argumentos.db) as conexion:
        print("AUDITORIA DEL ALMACEN f1.sqlite")
        auditar_integridad(conexion)
        auditar_nulos(conexion)
        auditar_anomalias(conexion)
        auditar_cobertura(conexion)
        auditar_era_elegida(conexion)

    reporte = salida.getvalue()
    print(reporte)
    RUTA_REPORTE.parent.mkdir(parents=True, exist_ok=True)
    RUTA_REPORTE.write_text(reporte, encoding="utf-8")
    print(f"Reporte guardado en {RUTA_REPORTE.relative_to(RAIZ)}")


if __name__ == "__main__":
    main()
