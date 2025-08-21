import streamlit as st
import pandas as pd
import numpy as np
from io import StringIO
import statsmodels
from statsmodels.stats.proportion import proportions_ztest
import altair as alt
import matplotlib.pyplot as plt
st.set_page_config(page_title="COVID-19 Viz – Pregunta 2", layout="wide")

GITHUB_BASE = "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_daily_reports"

@st.cache_data(show_spinner=False)
def load_daily_report(yyyy_mm_dd: str):
    yyyy, mm, dd = yyyy_mm_dd.split("-")
    url = f"{GITHUB_BASE}/{mm}-{dd}-{yyyy}.csv"
    df = pd.read_csv(url)
    # normalizar nombres por si varían
    lower = {c.lower(): c for c in df.columns}
    cols = {
        "country": lower.get("country_region", "Country_Region"),
        "province": lower.get("province_state", "Province_State"),
        "confirmed": lower.get("confirmed", "Confirmed"),
        "deaths": lower.get("deaths", "Deaths"),
        "recovered": lower.get("recovered", "Recovered") if "recovered" in lower else None,
        "active": lower.get("active", "Active") if "active" in lower else None,
    }
    return df, url, cols

st.sidebar.title("Opciones")
fecha = st.sidebar.date_input("Fecha del reporte (JHU CSSE)", value=pd.to_datetime("2022-09-09"))
fecha_str = pd.to_datetime(fecha).strftime("%Y-%m-%d")
df, source_url, cols = load_daily_report(fecha_str)
st.sidebar.caption(f"Fuente: {source_url}")

st.title("Exploración COVID-19 – Versión Streamlit (Preg2)")
st.caption("Adaptación fiel del script original: mostrar/ocultar filas/columnas y varios gráficos (líneas, barras, sectores, histograma y boxplot).")

# ———————————————————————————————————————————————
# a) Mostrar todas las filas del dataset, luego volver al estado inicial
# ———————————————————————————————————————————————
st.header("a) Mostrar filas")
mostrar_todas = st.checkbox("Mostrar todas las filas", value=False)
if mostrar_todas:
    st.dataframe(df, use_container_width=True)
else:
    st.dataframe(df.head(25), use_container_width=True)

# ———————————————————————————————————————————————
# b) Mostrar todas las columnas del dataset
# ———————————————————————————————————————————————
st.header("b) Mostrar columnas")
with st.expander("Vista de columnas"):
    st.write(list(df.columns))

st.caption("Usa el scroll horizontal de la tabla para ver todas las columnas en pantalla.")

# ———————————————————————————————————————————————
# c) Línea del total de fallecidos (>2500) vs Confirmed/Recovered/Active por país
# ———————————————————————————————————————————————
st.header("c) Gráfica de líneas por país (muertes > 2500)")
C, D = cols["confirmed"], cols["deaths"]
R, A = cols["recovered"], cols["active"]

metrics = [m for m in [C, D, R, A] if m and m in df.columns]
base = df[[cols["country"]] + metrics].copy()
base = base.rename(columns={cols["country"]: "Country_Region"})

filtrado = base.loc[base[D] > 2500]
agr = filtrado.groupby("Country_Region").sum(numeric_only=True)
orden = agr.sort_values(D)

if not orden.empty:
    st.line_chart(orden[[c for c in [C, R, A] if c in orden.columns]])

# ———————————————————————————————————————————————
# d) Barras de fallecidos de estados de Estados Unidos
# ———————————————————————————————————————————————
st.header("d) Barras: fallecidos por estado de EE.UU.")
country_col = cols["country"]
prov_col = cols["province"]

dfu = df[df[country_col] == "US"]
if len(dfu) == 0:
    st.info("Para esta fecha no hay registros con Country_Region='US'.")
else:
    agg_us = dfu.groupby(prov_col)[D].sum(numeric_only=True).sort_values(ascending=False)
    top_n = st.slider("Top estados por fallecidos", 5, 50, 20)
    st.bar_chart(agg_us.head(top_n))

# ———————————————————————————————————————————————
# e) Gráfica de sectores (simulada con barra si no hay pie nativo)
# ———————————————————————————————————————————————
st.header("e) Gráfica de sectores (simulada)")
lista_paises = ["Colombia", "Chile", "Peru", "Argentina", "Mexico"]
sel = st.multiselect("Países", sorted(df[country_col].unique().tolist()), default=lista_paises)
agg_latam = df[df[country_col].isin(sel)].groupby(country_col)[D].sum(numeric_only=True)
if agg_latam.sum() > 0:
    st.write("Participación de fallecidos")
    st.dataframe(agg_latam)
    # Como Streamlit no tiene pie nativo, mostramos distribución normalizada como barra
    normalized = agg_latam / agg_latam.sum()
    st.bar_chart(normalized)
else:
    st.warning("Sin datos para los países seleccionados")

# ———————————————————————————————————————————————
# f) Histograma del total de fallecidos por país (simulado con bar_chart)
# ———————————————————————————————————————————————
st.header("f) Histograma de fallecidos por país")
muertes_pais = df.groupby(country_col)[D].sum(numeric_only=True)
st.bar_chart(muertes_pais)

# ———————————————————————————————————————————————
# g) Boxplot de Confirmed, Deaths, Recovered, Active (simulado con box_chart)
# ———————————————————————————————————————————————
st.header("g) Boxplot (simulado)")
cols_box = [c for c in [C, D, R, A] if c and c in df.columns]
subset = df[cols_box].fillna(0)
subset_plot = subset.head(25)
# Streamlit no tiene boxplot nativo, así que mostramos estadísticas resumen en tabla
st.write("Resumen estadístico (simulación de boxplot):")
st.dataframe(subset_plot.describe().T)

# ———————————————————————————————————————————————
# 2.1. Estadística descriptiva: métricas por país
# ———————————————————————————————————————————————
st.header("2.1 Métricas clave por país")

# Necesitamos Confirmed y Deaths
agg_country = df.groupby(country_col)[[C, D]].sum(numeric_only=True)
agg_country["CFR"] = agg_country[D] / agg_country[C]

# Si tuvieras población, aquí podrías unirla. Por ahora simulamos con random:
agg_country["Tasa_100k"] = (agg_country[D] / (agg_country[C] + 1)) * 100000

st.dataframe(agg_country.head(20))

# ———————————————————————————————————————————————
# 2.2. Intervalos de confianza para CFR
# ———————————————————————————————————————————————
st.header("2.2 Intervalos de confianza (95%) del CFR")

def ci_cfr(deaths, confirmed, alpha=0.05):
    if confirmed == 0: return (0, 0)
    p = deaths / confirmed
    se = np.sqrt(p*(1-p)/confirmed)
    z = 1.96  # 95%
    return (p - z*se, p + z*se)

agg_country["CFR_IC"] = agg_country.apply(lambda row: ci_cfr(row[D], row[C]), axis=1)

st.dataframe(agg_country.head(20))

st.header("2.3 Test de hipótesis de CFR entre dos países")

paises = st.multiselect("Selecciona dos países", agg_country.index.tolist(), default=["Peru","Chile"])

if len(paises) == 2:
    deaths = [agg_country.loc[p, D] for p in paises]
    confirmed = [agg_country.loc[p, C] for p in paises]

    # Evitar división por cero
    if 0 in confirmed:
        st.error("Uno de los países seleccionados tiene 0 confirmados. No se puede calcular el test.")
    else:
        stat, pval = proportions_ztest(count=deaths, nobs=confirmed)
        st.write(f"Estadístico Z: {stat:.3f}, p-valor: {pval:.4f}")
        if pval < 0.05:
            st.success("Se rechaza H0: Hay diferencia significativa en CFR.")
        else:
            st.info("No se rechaza H0: No hay diferencia significativa en CFR.")

# ———————————————————————————————————————————————
# 2.4 Detección de outliers
# ———————————————————————————————————————————————
st.header("2.4 Outliers en fallecidos (Z-score)")

muertes_pais = df.groupby(country_col)[D].sum(numeric_only=True)
z_scores = (muertes_pais - muertes_pais.mean())/muertes_pais.std()
outliers = muertes_pais[z_scores > 3]

st.write("Outliers detectados (Z > 3):")
st.dataframe(outliers)


import matplotlib.pyplot as plt
# ———————————————————————————————————————————————
# 2.5 Gráfico de control (3σ) – Muertes diarias globales
# ———————————————————————————————————————————————
st.header("2.5 Gráfico de control (3σ) – Muertes diarias globales")

# Filtro: días hacia atrás
dias = st.sidebar.slider("Rango de días para análisis (2.5)", min_value=7, max_value=90, value=30, step=1)

# Construcción de rango de fechas dinámico
rango_fechas = pd.date_range(
    pd.to_datetime(fecha) - pd.Timedelta(days=dias),
    pd.to_datetime(fecha)
)

diario = []
for f in rango_fechas:
    try:
        df_tmp, _, cols_tmp = load_daily_report(f.strftime("%Y-%m-%d"))
        diario.append([f, df_tmp[cols_tmp["deaths"]].sum()])
    except:
        pass

serie = pd.DataFrame(diario, columns=["Fecha","Muertes"])

if not serie.empty:
    media = serie["Muertes"].mean()
    sigma = serie["Muertes"].std()
    ucl = media + 3*sigma
    lcl = max(0, media - 3*sigma)

    fig, ax = plt.subplots(figsize=(10,5))
    ax.plot(serie["Fecha"], serie["Muertes"], marker="o", label="Muertes diarias")
    ax.axhline(media, color="blue", linestyle="--", linewidth=2, label=f"Media ({media:.0f})")
    ax.axhline(ucl, color="red", linestyle="--", linewidth=2, label=f"UCL (+3σ={ucl:.0f})")
    ax.axhline(lcl, color="green", linestyle="--", linewidth=2, label=f"LCL (-3σ={lcl:.0f})")

    # resaltar puntos fuera de control
    fuera_control = serie[(serie["Muertes"] > ucl) | (serie["Muertes"] < lcl)]
    ax.scatter(fuera_control["Fecha"], fuera_control["Muertes"], color="red", s=80, zorder=5, label="Anomalías")

    ax.set_title(f"Gráfico de Control (3σ) – Últimos {dias} días", fontsize=14)
    ax.set_xlabel("Fecha")
    ax.set_ylabel("Muertes")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.6)

    st.pyplot(fig)

    st.write(f"Media: {media:.1f}, UCL (Límite superior 3σ): {ucl:.1f}, LCL: {lcl:.1f}")
else:
    st.warning("No se pudo construir la serie de muertes para este rango de fechas.")


