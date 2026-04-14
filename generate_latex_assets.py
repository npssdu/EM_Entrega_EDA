from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


def zscore_matrix(X: np.ndarray) -> np.ndarray:
        """Estandariza por columnas (media 0, sd 1) usando sd muestral (ddof=1)."""

        mu = X.mean(axis=0)
        sigma = X.std(axis=0, ddof=1)
        sigma = np.where(sigma == 0, 1.0, sigma)
        return (X - mu) / sigma


def pca_from_zscored(Xz: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """PCA vía SVD.

        Returns:
            - explained_variance_ratio: (n_components,)
            - loadings: (n_features, n_components) (equivalente a rotation en prcomp)
        """

        # Xz: (n_samples, n_features)
        n_samples = Xz.shape[0]
        U, S, Vt = np.linalg.svd(Xz, full_matrices=False)
        explained_variance = (S**2) / (n_samples - 1)
        explained_variance_ratio = explained_variance / explained_variance.sum()
        loadings = Vt.T
        return explained_variance_ratio, loadings


def _make_names(columns: list[str]) -> list[str]:
    """R-like make.names() for the subset of cases we have here.

    - Replaces spaces with '.' to match the Rmd naming (e.g., 'concave points_mean' -> 'concave.points_mean').
    - Leaves other characters untouched (dataset already uses safe names).
    """

    return [c.replace(" ", ".") for c in columns]


def _safe_filename(name: str) -> str:
    return (
        name.replace(" ", "_")
        .replace(".", "_")
        .replace("/", "_")
        .replace("\\", "_")
    )


def cohen_d(x: pd.Series, g: pd.Series, pos: str = "M", neg: str = "B") -> float:
    x1 = x[g == pos].astype(float)
    x0 = x[g == neg].astype(float)
    n1 = x1.shape[0]
    n0 = x0.shape[0]
    s1 = x1.std(ddof=1)
    s0 = x0.std(ddof=1)
    s_pooled = math.sqrt(((n1 - 1) * (s1**2) + (n0 - 1) * (s0**2)) / (n1 + n0 - 2))
    return float((x1.mean() - x0.mean()) / s_pooled)


def save_table(df: pd.DataFrame, out_path: Path, float_format: str = "%.4f") -> None:
    out_path.write_text(
        df.to_latex(
            index=False,
            escape=True,
            float_format=float_format,
        ),
        encoding="utf-8",
    )


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    csv_path = base_dir / "Cancer_Data.csv"

    figs_dir = base_dir / "docuemento" / "figs"
    tables_dir = base_dir / "docuemento" / "tables"
    figs_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    # -------------------- Load + clean --------------------
    data = pd.read_csv(csv_path)
    data.columns = _make_names(list(data.columns))

    # Remove residual empty column (the CSV ends with a trailing comma in header)
    data = data.dropna(axis=1, how="all")

    # Diagnosis as category
    if "diagnosis" not in data.columns:
        raise ValueError("No se encontró la columna 'diagnosis' en Cancer_Data.csv")

    data["diagnosis"] = data["diagnosis"].astype("category")

    # Keep only *_mean (plus diagnosis)
    mean_cols = [c for c in data.columns if c.endswith("_mean")]
    data_clean = data[["diagnosis", *mean_cols]].drop_duplicates().copy()

    # Numeric block
    data_num = data_clean.drop(columns=["diagnosis"]).apply(pd.to_numeric, errors="coerce")

    # -------------------- Basic checks (optional tables) --------------------
    counts = (
        data_clean["diagnosis"].value_counts().rename_axis("diagnosis").reset_index(name="n")
    )
    save_table(counts, tables_dir / "tabla_conteos_diagnostico.tex", float_format="%.0f")

    # -------------------- 3. Univariate: histogram + boxplot --------------------
    sns.set_theme(style="whitegrid")

    univar_vars = [
        "radius_mean",
        "texture_mean",
        "smoothness_mean",
        "compactness_mean",
        "concavity_mean",
        "concave.points_mean",
        "symmetry_mean",
        "fractal_dimension_mean",
    ]

    for var in univar_vars:
        if var not in data_clean.columns:
            raise ValueError(f"No se encontró la variable requerida: {var}")

        fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)

        # Histogram
        sns.histplot(
            data_clean[var].astype(float),
            bins=30,
            ax=axes[0],
            color="steelblue",
            edgecolor="black",
            alpha=0.7,
        )
        axes[0].set_title(f"Histograma de {var}")
        axes[0].set_xlabel(var)
        axes[0].set_ylabel("Frecuencia")

        # Boxplot by diagnosis
        sns.boxplot(
            data=data_clean,
            x="diagnosis",
            y=var,
            ax=axes[1],
            palette={"B": "lightgreen", "M": "salmon"},
        )
        axes[1].set_title(f"Boxplot de {var} por diagnóstico")
        axes[1].set_xlabel("Diagnóstico")
        axes[1].set_ylabel(var)

        out_path = figs_dir / f"univar_{_safe_filename(var)}.png"
        fig.savefig(out_path, dpi=300)
        plt.close(fig)

    # Descriptive summary table (Media/Mediana/Desviación) for all mean vars
    medidas_resumen = pd.DataFrame(
        {
            "Variable": data_num.columns,
            "Media": data_num.mean(axis=0).values,
            "Mediana": data_num.median(axis=0).values,
            "Desviacion": data_num.std(axis=0, ddof=1).values,
        }
    )
    save_table(medidas_resumen, tables_dir / "tabla_medidas_resumen.tex")

    # -------------------- 4. Bivariate: correlation matrix + scatter matrix --------------------
    data_num_reducido = data_num.drop(columns=["radius_mean", "perimeter_mean"], errors="ignore")

    mat_cor = data_num_reducido.corr(method="pearson")

    fig, ax = plt.subplots(figsize=(10, 9), constrained_layout=True)
    mask = np.tril(np.ones_like(mat_cor, dtype=bool))
    sns.heatmap(
        mat_cor,
        mask=mask,
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        annot=True,
        fmt=".2f",
        linewidths=0.5,
        square=True,
        cbar_kws={"label": "r de Pearson"},
        ax=ax,
    )
    ax.set_title("Matriz de correlación (variables *_mean reducidas)")
    fig.savefig(figs_dir / "corr_matrix.png", dpi=300)
    plt.close(fig)

    # Scatterplot matrix (pairplot)
    pair_df = data_clean[["diagnosis", *list(data_num_reducido.columns)]].copy()
    g = sns.pairplot(
        pair_df,
        hue="diagnosis",
        corner=True,
        plot_kws={"alpha": 0.6, "s": 15, "edgecolor": "none"},
        diag_kind="hist",
        diag_kws={"bins": 20, "alpha": 0.7},
        palette={"B": "steelblue", "M": "tomato"},
    )
    g.fig.suptitle("Matriz de scatterplots (variables *_mean reducidas)", y=1.02)
    g.fig.savefig(figs_dir / "scatter_matrix.png", dpi=300, bbox_inches="tight")
    plt.close(g.fig)

    # -------------------- 6. Parallel coordinates (z-score) --------------------
    z = zscore_matrix(data_num.values)
    z_df = pd.DataFrame(z, columns=data_num.columns)
    z_df.insert(0, "diagnosis", data_clean["diagnosis"].astype(str).values)

    # Matplotlib parallel coordinates
    from pandas.plotting import parallel_coordinates

    fig, ax = plt.subplots(figsize=(14, 6), constrained_layout=True)
    parallel_coordinates(
        z_df,
        class_column="diagnosis",
        color=("steelblue", "tomato"),
        alpha=0.25,
        ax=ax,
    )
    ax.set_title("Coordenadas paralelas (Z-score)")
    ax.set_xlabel("Variable")
    ax.set_ylabel("Puntuación Z")
    ax.tick_params(axis="x", rotation=90)
    fig.savefig(figs_dir / "parallel_coordinates.png", dpi=300)
    plt.close(fig)

    # -------------------- 6. 3D plots (static) + stats table --------------------
    plot3d_specs = [
        ("radius_mean", "smoothness_mean", "concavity_mean", "Gráfica 1", "radius_mean", "concavity_mean"),
        ("radius_mean", "compactness_mean", "concavity_mean", "Gráfica 2", "compactness_mean", "concavity_mean"),
        ("texture_mean", "smoothness_mean", "compactness_mean", "Gráfica 3", "texture_mean", "smoothness_mean"),
        ("texture_mean", "smoothness_mean", "concavity_mean", "Gráfica 4", "texture_mean", "concavity_mean"),
        ("texture_mean", "smoothness_mean", "symmetry_mean", "Gráfica 5", "smoothness_mean", "symmetry_mean"),
        ("smoothness_mean", "compactness_mean", "concavity_mean", "Gráfica 6", "compactness_mean", "concavity_mean"),
        ("smoothness_mean", "compactness_mean", "symmetry_mean", "Gráfica 7", "smoothness_mean", "symmetry_mean"),
        ("compactness_mean", "concavity_mean", "symmetry_mean", "Gráfica 8", "concavity_mean", "symmetry_mean"),
    ]

    stats_rows: list[dict[str, object]] = []

    for idx, (x, y, zvar, titulo, v1, v2) in enumerate(plot3d_specs, start=1):
        fig = plt.figure(figsize=(9, 7), constrained_layout=True)
        ax = fig.add_subplot(111, projection="3d")

        for diag, color in [("B", "steelblue"), ("M", "tomato")]:
            sub = data_clean[data_clean["diagnosis"].astype(str) == diag]
            ax.scatter(
                sub[x].astype(float),
                sub[y].astype(float),
                sub[zvar].astype(float),
                s=18,
                alpha=0.7,
                c=color,
                edgecolors="k",
                linewidths=0.2,
                label=diag,
            )

        ax.set_title(f"{titulo}: {x} vs {y} vs {zvar}")
        ax.set_xlabel(x)
        ax.set_ylabel(y)
        ax.set_zlabel(zvar)
        ax.legend(title="diagnosis")

        fig.savefig(figs_dir / f"plot3d_{idx}.png", dpi=300)
        plt.close(fig)

        # Stats: Pearson r + slopes per class (v2 ~ v1)
        v1_all = data_clean[v1].astype(float)
        v2_all = data_clean[v2].astype(float)
        r = float(np.corrcoef(v1_all, v2_all)[0, 1])

        slopes = {}
        for diag in ["B", "M"]:
            sub = data_clean[data_clean["diagnosis"].astype(str) == diag]
            xv = sub[v1].astype(float).values
            yv = sub[v2].astype(float).values
            # simple linear regression slope
            slope = float(np.polyfit(xv, yv, deg=1)[0])
            slopes[diag] = slope

        stats_rows.append(
            {
                "Grafica": titulo,
                "v1": v1,
                "v2": v2,
                "r_Pearson": r,
                "Pendiente_B": slopes["B"],
                "Pendiente_M": slopes["M"],
            }
        )

    stats_3d = pd.DataFrame(stats_rows)
    save_table(stats_3d, tables_dir / "tabla_stats_3d.tex")

    # -------------------- 7. Multivariate descriptive: mean vector barplot --------------------
    media_vectorial = data_num.mean(axis=0)
    df_media = pd.DataFrame({"Variable": media_vectorial.index, "Media": media_vectorial.values})
    df_media = df_media.sort_values("Media", ascending=True)

    fig, ax = plt.subplots(figsize=(10, 7), constrained_layout=True)
    ax.barh(df_media["Variable"], df_media["Media"], color="dodgerblue", alpha=0.85)
    ax.set_xscale("log")
    ax.set_title("Media vectorial (escala log10)")
    ax.set_xlabel("Media (escala log)")
    ax.set_ylabel("Variable")
    fig.savefig(figs_dir / "mean_vector.png", dpi=300)
    plt.close(fig)

    # -------------------- 7. Covariance heatmap (log10(|cov|)) --------------------
    mat_cov = np.cov(data_num.values, rowvar=False, ddof=1)
    cov_df = pd.DataFrame(mat_cov, index=data_num.columns, columns=data_num.columns)

    cov_log = np.log10(np.abs(cov_df) + 1e-5)

    fig, ax = plt.subplots(figsize=(10, 9), constrained_layout=True)
    sns.heatmap(
        cov_log,
        cmap="magma",
        linewidths=0.5,
        square=True,
        cbar_kws={"label": "Log10(|Cov|)"},
        ax=ax,
    )
    ax.set_title("Mapa de calor de covarianzas (log10(|Cov|))")
    ax.tick_params(axis="x", rotation=90)
    fig.savefig(figs_dir / "cov_heatmap.png", dpi=300)
    plt.close(fig)

    # -------------------- 8. Global interpretation support tables --------------------
    # Correlations >= 0.75 (unique pairs)
    mat_cor_global = data_num.corr(method="pearson")
    cor_pairs = []
    cols = list(mat_cor_global.columns)
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            v1 = cols[i]
            v2 = cols[j]
            cor_val = float(mat_cor_global.loc[v1, v2])
            if abs(cor_val) >= 0.75:
                cor_pairs.append({"Var1": v1, "Var2": v2, "Cor": cor_val, "AbsCor": abs(cor_val)})

    cor_altas = pd.DataFrame(cor_pairs).sort_values("AbsCor", ascending=False)
    save_table(cor_altas.head(10).drop(columns=["AbsCor"]), tables_dir / "tabla_cor_altas.tex")

    # Cohen's d per variable
    efectos = []
    for c in data_num.columns:
        d = cohen_d(data_num[c], data_clean["diagnosis"].astype(str))
        efectos.append({"variable": c, "cohen_d": d, "abs_d": abs(d)})

    tabla_efectos = pd.DataFrame(efectos).sort_values("abs_d", ascending=False)
    save_table(tabla_efectos.head(10).drop(columns=["abs_d"]), tables_dir / "tabla_efectos_cohen.tex")

    # PCA (standardized)
    Xz = zscore_matrix(data_num.values)
    var_exp, loadings = pca_from_zscored(Xz)
    var_acum = np.cumsum(var_exp)
    tabla_pca = pd.DataFrame(
        {
            "Componente": [f"PC{i+1}" for i in range(len(var_exp))],
            "Varianza": var_exp,
            "Varianza_Acumulada": var_acum,
        }
    )
    save_table(tabla_pca.head(5), tables_dir / "tabla_pca_varianza.tex")

    # PC1 loadings
    loadings_pc1 = pd.DataFrame(
        {
            "variable": list(data_num.columns),
            "loading_pc1": loadings[:, 0],
            "abs_loading_pc1": np.abs(loadings[:, 0]),
        }
    ).sort_values("abs_loading_pc1", ascending=False)

    save_table(loadings_pc1.head(10).drop(columns=["abs_loading_pc1"]), tables_dir / "tabla_cargas_pc1.tex")

    print("OK: Figuras y tablas generadas en docuemento/figs y docuemento/tables")


if __name__ == "__main__":
    main()
