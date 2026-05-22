"""
Práctica 4 - IA: Clasificación Académica
Paso 1: Análisis Exploratorio de Datos (EDA)

Dataset: Wine Quality (UCI Machine Learning Repository)
Fuente: https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data

Contiene 178 muestras de vinos italianos clasificados en 3 tipos (clases 1, 2, 3),
con 13 atributos químicos cada uno.
"""

import io
import ssl
import urllib.request

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional

# ---------------------------------------------------------------------------
# Constantes del dataset
# ---------------------------------------------------------------------------

DATASET_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data"
)

COLUMN_NAMES: list[str] = [
    "class",
    "alcohol",
    "malic_acid",
    "ash",
    "alcalinity_of_ash",
    "magnesium",
    "total_phenols",
    "flavanoids",
    "nonflavanoid_phenols",
    "proanthocyanins",
    "color_intensity",
    "hue",
    "od280_od315",
    "proline",
]

TARGET_COL = "class"


# ---------------------------------------------------------------------------
# Carga del dataset
# ---------------------------------------------------------------------------

def load_dataset(url: str, column_names: list[str]) -> pd.DataFrame:
    """
    Descarga y carga el dataset desde una URL pública.

    En macOS con Python ≥ 3.12 el bundle de certificados del sistema puede
    no estar vinculado; se usa un contexto SSL sin verificación sólo para
    este descargue puntual de un recurso académico conocido.

    Args:
        url: URL directa al archivo CSV del repositorio UCI.
        column_names: Lista de nombres para las columnas (incluye la clase).

    Returns:
        DataFrame con los datos cargados y la columna de clase como string.
    """
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE

    with urllib.request.urlopen(url, context=ctx) as response:
        raw_bytes = response.read()

    df = pd.read_csv(io.BytesIO(raw_bytes), header=None, names=column_names)
    # Convertir clase a string para que seaborn la trate como variable categórica
    df[TARGET_COL] = df[TARGET_COL].astype(str).apply(lambda x: f"Clase {x}")
    return df


# ---------------------------------------------------------------------------
# Información básica
# ---------------------------------------------------------------------------

def print_basic_info(df: pd.DataFrame) -> None:
    """Imprime dimensiones y primeras filas del dataset."""
    separator = "=" * 60
    print(f"\n{separator}")
    print("  INFORMACIÓN BÁSICA DEL DATASET")
    print(separator)
    print(f"  Filas    : {df.shape[0]}")
    print(f"  Columnas : {df.shape[1]}  (1 clase + {df.shape[1] - 1} atributos)")
    print(f"\n  Primeras 5 filas:\n")
    print(df.head().to_string(index=True))


# ---------------------------------------------------------------------------
# Distribución de clases
# ---------------------------------------------------------------------------

def print_class_distribution(df: pd.DataFrame) -> None:
    """
    Imprime cuántas muestras pertenecen a cada clase y su porcentaje
    respecto al total.
    """
    separator = "=" * 60
    print(f"\n{separator}")
    print("  DISTRIBUCIÓN DE CLASES")
    print(separator)

    counts: pd.Series = df[TARGET_COL].value_counts().sort_index()
    total: int = len(df)

    for cls, count in counts.items():
        pct = count / total * 100
        bar = "█" * int(pct / 2)          # barra visual proporcional
        print(f"  {cls}: {count:>3} muestras  ({pct:.1f}%)  {bar}")


# ---------------------------------------------------------------------------
# Estadísticas por atributo
# ---------------------------------------------------------------------------

def print_statistics(df: pd.DataFrame) -> None:
    """
    Calcula e imprime media y desviación estándar de cada atributo numérico,
    usando numpy para asegurar el cálculo desde cero (sin scikit-learn).
    """
    separator = "=" * 60
    print(f"\n{separator}")
    print("  ESTADÍSTICAS POR ATRIBUTO  (Media ± Desviación Estándar)")
    print(separator)

    feature_cols: list[str] = [c for c in df.columns if c != TARGET_COL]

    # Encabezado de tabla
    print(f"  {'Atributo':<25} {'Media':>12} {'Desv. Est.':>12}")
    print(f"  {'-'*25} {'-'*12} {'-'*12}")

    for col in feature_cols:
        values: np.ndarray = df[col].to_numpy(dtype=float)
        mean: float = float(np.mean(values))
        std: float = float(np.std(values, ddof=1))   # ddof=1 → muestral
        print(f"  {col:<25} {mean:>12.4f} {std:>12.4f}")


# ---------------------------------------------------------------------------
# Scatter plots 2D por pares de atributos
# ---------------------------------------------------------------------------

def plot_scatter_pairs(
    df: pd.DataFrame,
    output_path: Optional[str] = "scatter_plots.png",
) -> None:
    """
    Genera un pair-plot (scatter matrix) coloreando cada punto según su clase.
    Sólo se grafican los 6 primeros atributos para mantener legibilidad.

    Args:
        df: DataFrame con atributos y columna de clase.
        output_path: Ruta donde guardar la imagen. Si es None, muestra en pantalla.
    """
    feature_cols: list[str] = [c for c in df.columns if c != TARGET_COL]
    # Limitamos a los 6 primeros atributos para mayor legibilidad
    selected: list[str] = feature_cols[:6]

    palette = {
        "Clase 1": "#E63946",
        "Clase 2": "#457B9D",
        "Clase 3": "#2A9D8F",
    }

    plot_df = df[selected + [TARGET_COL]].copy()

    grid = sns.pairplot(
        plot_df,
        hue=TARGET_COL,
        palette=palette,
        diag_kind="kde",
        plot_kws={"alpha": 0.65, "s": 40, "edgecolors": "white", "linewidth": 0.3},
        diag_kws={"fill": True, "alpha": 0.4},
    )

    grid.figure.suptitle(
        "Scatter Plots 2D — Dataset Wine (UCI)\nPrimeros 6 atributos",
        y=1.02,
        fontsize=13,
        fontweight="bold",
    )

    if output_path:
        grid.savefig(output_path, bbox_inches="tight", dpi=150)
        print(f"\n  Gráfica guardada en: '{output_path}'")
    else:
        plt.show()

    plt.close("all")


def plot_class_boxplots(
    df: pd.DataFrame,
    output_path: Optional[str] = "boxplots.png",
) -> None:
    """
    Genera boxplots de todos los atributos agrupados por clase para detectar
    diferencias de distribución entre grupos.

    Args:
        df: DataFrame con atributos y columna de clase.
        output_path: Ruta donde guardar la imagen. Si es None, muestra en pantalla.
    """
    feature_cols: list[str] = [c for c in df.columns if c != TARGET_COL]
    n_features = len(feature_cols)

    fig, axes = plt.subplots(
        nrows=3, ncols=5, figsize=(20, 11), constrained_layout=True
    )
    axes_flat = axes.flatten()

    palette = ["#E63946", "#457B9D", "#2A9D8F"]

    for i, col in enumerate(feature_cols):
        ax = axes_flat[i]
        sns.boxplot(
            data=df,
            x=TARGET_COL,
            y=col,
            hue=TARGET_COL,
            palette=palette,
            legend=False,
            ax=ax,
            linewidth=1.2,
            flierprops={"marker": "o", "markersize": 3, "alpha": 0.5},
        )
        ax.set_title(col, fontsize=9, fontweight="bold")
        ax.set_xlabel("")
        ax.set_ylabel("")

    # Ocultar subplot sobrante (13 features → 15 celdas)
    for j in range(n_features, len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.suptitle(
        "Distribución por Clase — Dataset Wine (UCI)",
        fontsize=14,
        fontweight="bold",
    )

    if output_path:
        fig.savefig(output_path, bbox_inches="tight", dpi=150)
        print(f"  Gráfica guardada en: '{output_path}'")
    else:
        plt.show()

    plt.close("all")


# ---------------------------------------------------------------------------
# Punto de entrada
# ---------------------------------------------------------------------------

def main() -> None:
    print("\n  Cargando dataset Wine desde el repositorio UCI...")
    df: pd.DataFrame = load_dataset(DATASET_URL, COLUMN_NAMES)

    print_basic_info(df)
    print_class_distribution(df)
    print_statistics(df)

    print("\n  Generando visualizaciones...")
    plot_scatter_pairs(df, output_path="scatter_plots.png")
    plot_class_boxplots(df, output_path="boxplots.png")

    print("\n  EDA completado.\n")


if __name__ == "__main__":
    main()
