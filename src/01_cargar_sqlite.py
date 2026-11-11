"""Fase 1 - Almacen transaccional.

Carga los 14 archivos CSV del dataset "Formula 1 World Championship
(1950 - 2024)" de Kaggle en una sola base de datos SQLite (data/f1.sqlite)
y muestra un inventario de lo que quedo cargado.

Uso (desde la carpeta F1):
    python src/01_cargar_sqlite.py
"""

from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path

import pandas as pd

RAIZ: Path = Path(__file__).resolve().parent.parent
CARPETA_CSV: Path = RAIZ / "data" / "raw"
RUTA_DB: Path = RAIZ / "data" / "f1.sqlite"

# En este dataset los valores faltantes vienen escritos como \N
MARCA_NULO: str = "\\N"


def cargar_csv(ruta_csv: Path) -> pd.DataFrame:
    """Lee un CSV y convierte la marca \\N en nulos reales."""
    tabla = pd.read_csv(ruta_csv, na_values=[MARCA_NULO], keep_default_na=False)
    # Evita que una columna entera con nulos se guarde como decimal (1.0, 2.0...)
    return tabla.convert_dtypes()


def construir_almacen(carpeta_csv: Path, ruta_db: Path) -> list[tuple[str, int, int]]:
    """Crea una tabla por cada CSV y devuelve (tabla, filas, columnas)."""
    archivos = sorted(carpeta_csv.glob("*.csv"))
    if not archivos:
        raise FileNotFoundError(f"No hay archivos CSV en {carpeta_csv}")

    inventario: list[tuple[str, int, int]] = []
    with sqlite3.connect(ruta_db) as conexion:
        for ruta_csv in archivos:
            nombre_tabla = ruta_csv.stem
            tabla = cargar_csv(ruta_csv)
            tabla.to_sql(nombre_tabla, conexion, if_exists="replace", index=False)
            inventario.append((nombre_tabla, len(tabla), tabla.shape[1]))
    return inventario


def imprimir_inventario(inventario: list[tuple[str, int, int]], ruta_db: Path) -> None:
    """Muestra el resumen del almacen: tablas, filas, columnas y peso."""
    print("ALMACEN TRANSACCIONAL - f1.sqlite")
    print("-" * 48)
    print(f"{'Tabla':<24}{'Filas':>12}{'Columnas':>12}")
    print("-" * 48)
    for nombre, filas, columnas in inventario:
        print(f"{nombre:<24}{filas:>12,}{columnas:>12}")
    print("-" * 48)
    total_filas = sum(filas for _, filas, _ in inventario)
    print(f"{'TOTAL':<24}{total_filas:>12,}")
    print(f"Tablas: {len(inventario)}")
    print(f"Peso de la base de datos: {ruta_db.stat().st_size / 1_000_000:.1f} MB")

    with sqlite3.connect(ruta_db) as conexion:
        primera, ultima, carreras = conexion.execute(
            "SELECT MIN(year), MAX(year), COUNT(*) FROM races"
        ).fetchone()
    print(f"Temporadas: {primera} a {ultima} ({carreras:,} carreras)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Carga los CSV de F1 a SQLite.")
    parser.add_argument("--db", type=Path, default=RUTA_DB, help="Ruta del archivo SQLite de salida")
    argumentos = parser.parse_args()

    inventario = construir_almacen(CARPETA_CSV, argumentos.db)
    imprimir_inventario(inventario, argumentos.db)


if __name__ == "__main__":
    main()
