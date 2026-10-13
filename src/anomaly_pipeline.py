"""AgroAnomalie Tataouine - moteur d'anomalies (Sprints 1-2, CRISP-DM : preparation + modelisation).

Entree  : data/processed/climate_daily_clean.csv (Open-Meteo, quotidien, 1980-01-01 -> 2026-09-01)
Sorties : data/processed/climate_anomalies.csv, data/processed/spi_monthly.csv,
          reports/validation_summary.txt, reports/figures/*.png

Methode
  1. Preparation : humidite du sol clippee a 0, 2026 exclue des statistiques annuelles.
  2. Climatologie : moyenne et ecart-type par jour calendaire sur la periode de reference
     1991-2020 (norme OMM), fenetre circulaire de +/- 7 jours.
  3. Z-score : (valeur - normale du jour) / ecart-type du jour. Alerte chaleur si z >= 2.
  4. SPI : cumuls mensuels glissants (1, 3, 6, 12 mois), loi gamma ajustee par mois calendaire
     sur 1991-2020, avec traitement des mois a zero pluie (melange discret + gamma).
  5. Isolation Forest : anomalies multivariees sur les ecarts standardises.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import IsolationForest

ROOT = Path(__file__).resolve().parents[1]
REF = (1991, 2020)
WINDOW = 7  # jours de part et d'autre
SEED = 42
PROCESSED_DIR = ROOT / "data" / "processed"
REPORTS_DIR = ROOT / "reports"

# ----------------------------------------------------------------------------------------------
# 1. Chargement et preparation
# ----------------------------------------------------------------------------------------------
INPUT_FILE = (
    ROOT / "data" / "processed"
    / "climate_daily_clean.csv"
)

df = pd.read_csv(
    INPUT_FILE,
    parse_dates=["date"]
).set_index("date")

df["soil_moisture_0_7cm_mean_m3m3"] = df["soil_moisture_0_7cm_mean_m3m3"].clip(lower=0)
df["soil_moisture_7_28cm_mean_m3m3"] = df["soil_moisture_7_28cm_mean_m3m3"].clip(lower=0)

# jour calendaire sur 365 jours (le 29 fevrier est rattache au 28 fevrier)
idx = df.index
doy = idx.dayofyear.to_numpy().copy()
leap = idx.is_leap_year
doy = np.where(leap & (idx.month > 2), doy - 1, doy)
doy = np.where(leap & (idx.month == 2) & (idx.day == 29), 59, doy)
df["doy"] = doy

ref = df[(df.index.year >= REF[0]) & (df.index.year <= REF[1])]


def climatology(col: str) -> pd.DataFrame:
    """Moyenne et ecart-type par jour calendaire, fenetre circulaire +/- WINDOW jours."""
    by_doy = {d: g[col].to_numpy() for d, g in ref.groupby("doy")}
    out = []
    for d in range(1, 366):
        days = [((d - 1 + k) % 365) + 1 for k in range(-WINDOW, WINDOW + 1)]
        vals = np.concatenate([by_doy[x] for x in days])
        out.append((d, vals.mean(), vals.std(ddof=1)))
    return pd.DataFrame(out, columns=["doy", "mean", "std"]).set_index("doy")


VARS = {
    "tmax": "temperature_max_c",
    "tmean": "temperature_mean_c",
    "tmin": "temperature_min_c",
    "rh": "relative_humidity_mean_pct",
    "wind": "wind_speed_10m_mean_kmh",
    "soil": "soil_moisture_0_7cm_mean_m3m3",
}
clim = {k: climatology(c) for k, c in VARS.items()}

for k, c in VARS.items():
    m = df["doy"].map(clim[k]["mean"])
    s = df["doy"].map(clim[k]["std"])
    df[f"{k}_normal"] = m
    df[f"{k}_anom"] = df[c] - m
    df[f"z_{k}"] = (df[c] - m) / s

# ----------------------------------------------------------------------------------------------
# 2. Alertes chaleur (z-score) et episodes de canicule
# ----------------------------------------------------------------------------------------------
df["heat_flag"] = (df["z_tmax"] >= 2).astype(int)
df["heat_extreme_flag"] = (df["z_tmax"] >= 3).astype(int)
df["cold_flag"] = (df["z_tmin"] <= -2).astype(int)

run = (df["heat_flag"].diff().fillna(0) != 0).cumsum()
run_len = df.groupby(run)["heat_flag"].transform("sum") * df["heat_flag"]
df["heatwave_flag"] = (run_len >= 3).astype(int)  # >= 3 jours consecutifs a z >= 2

# ----------------------------------------------------------------------------------------------
# 3. SPI (1, 3, 6, 12 mois)
# ----------------------------------------------------------------------------------------------
monthly = df["rain_sum_mm"].resample("MS").sum()
monthly.loc[monthly.index > "2026-08-01"] = np.nan  # septembre 2026 incomplet


def spi_series(rain_m: pd.Series, scale: int) -> pd.Series:
    acc = rain_m.rolling(scale, min_periods=scale).sum()
    out = pd.Series(np.nan, index=acc.index)
    for month in range(1, 13):
        sel = acc[acc.index.month == month].dropna()
        ref_vals = sel[(sel.index.year >= REF[0]) & (sel.index.year <= REF[1])]
        q = (ref_vals == 0).mean()  # probabilite d'un cumul nul
        pos = ref_vals[ref_vals > 0]
        if len(pos) < 10:
            continue
        a, loc, b = stats.gamma.fit(pos, floc=0)
        cdf = q + (1 - q) * stats.gamma.cdf(sel, a, loc=0, scale=b)
        cdf = np.clip(cdf, 1e-4, 1 - 1e-4)
        out.loc[sel.index] = stats.norm.ppf(cdf)
    return out


spi = pd.DataFrame({f"spi{n}": spi_series(monthly, n) for n in (1, 3, 6, 12)})
spi.insert(0, "rain_mm", monthly)


def spi_class(x: float) -> str:
    if np.isnan(x):
        return ""
    if x <= -2:
        return "secheresse extreme"
    if x <= -1.5:
        return "secheresse severe"
    if x <= -1:
        return "secheresse moderee"
    if x >= 2:
        return "extremement humide"
    if x >= 1.5:
        return "tres humide"
    if x >= 1:
        return "humide"
    return "normal"


spi["spi12_class"] = spi["spi12"].map(spi_class)
spi["spi6_class"] = spi["spi6"].map(spi_class)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
spi.to_csv(PROCESSED_DIR / "spi_monthly.csv", float_format="%.3f")

# SPI mensuel recopie sur chaque jour (valeur du mois en cours)
month_key = df.index.to_period("M").to_timestamp()
for c in ("spi1", "spi3", "spi6", "spi12"):
    df[c] = month_key.map(spi[c])

# ----------------------------------------------------------------------------------------------
# 4. Isolation Forest (multivarie) sur les ecarts standardises
# ----------------------------------------------------------------------------------------------
feat = ["z_tmax", "z_tmean", "z_tmin", "z_rh", "z_wind", "z_soil"]
X = df[feat].copy()
X["rain_log"] = np.log1p(df["rain_sum_mm"])
iso = IsolationForest(n_estimators=300, contamination=0.01, random_state=SEED)
iso.fit(X[(X.index.year >= REF[0]) & (X.index.year <= REF[1])])
df["iforest_score"] = -iso.score_samples(X)  # plus grand = plus anormal
thr = np.quantile(df.loc[ref.index, "iforest_score"], 0.99)
df["iforest_flag"] = (df["iforest_score"] >= thr).astype(int)

keep = [
    "temperature_mean_c", "temperature_max_c", "temperature_min_c", "rain_sum_mm",
    "tmax_normal", "tmax_anom", "z_tmax", "z_tmean", "z_tmin", "z_rh", "z_wind", "z_soil",
    "heat_flag", "heat_extreme_flag", "heatwave_flag", "cold_flag",
    "spi1", "spi3", "spi6", "spi12", "iforest_score", "iforest_flag",
]
df[keep].to_csv(PROCESSED_DIR / "climate_anomalies.csv", float_format="%.4f")

# ----------------------------------------------------------------------------------------------
# 5. Validation sur des episodes connus + resume
# ----------------------------------------------------------------------------------------------
lines = []
P = lines.append
yr = df[df.index.year < 2026]
annual = yr.groupby(yr.index.year).agg(rain=("rain_sum_mm", "sum"), tmean=("temperature_mean_c", "mean"),
                                       hot=("hot_day_tmax_ge_40c", "sum"),
                                       heat_days=("heat_flag", "sum"), hw_days=("heatwave_flag", "sum"))
refann = annual.loc[REF[0]:REF[1]]
P(f"Normale {REF[0]}-{REF[1]} : pluie annuelle moyenne {refann.rain.mean():.1f} mm "
  f"(ecart-type {refann.rain.std():.1f}), Tmean {refann.tmean.mean():.2f} C")
P("")
P("SPI-12 en decembre (cumul de l'annee civile), annees avec SPI-12 <= -1 :")
d12 = spi.loc[spi.index.month == 12, ["rain_mm", "spi12"]].dropna()
d12.index = d12.index.year
ann_rain = annual["rain"]
for y, r in d12[d12.spi12 <= -1].iterrows():
    P(f"  {y}: pluie annuelle {ann_rain[y]:6.1f} mm | SPI-12 = {r.spi12:+.2f} ({spi_class(r.spi12)})")
P("")
P("Rang des annees les plus seches (pluie annuelle, 1980-2025) :")
for y, v in ann_rain.sort_values().head(8).items():
    P(f"  {y}: {v:6.1f} mm | SPI-12 dec = {d12.spi12.get(y, np.nan):+.2f}")
P("")
for a, b in [("2000-01-01", "2001-12-31"), ("2021-01-01", "2022-12-31")]:
    s = spi.loc[a:b, "spi12"].dropna()
    P(f"Periode {a[:4]}-{b[:4]} : SPI-12 min {s.min():+.2f} ({s.idxmin():%Y-%m}), "
      f"{(s <= -1).sum()} mois sur {len(s)} a SPI-12 <= -1")
P("")
P("Chaleur (z_tmax >= 2) : jours par annee, moyenne par decennie")
yr2 = annual.copy()
yr2["dec"] = (yr2.index // 10) * 10
for d, g in yr2.groupby("dec"):
    P(f"  {d}s : {g.heat_days.mean():5.1f} jours z>=2 | {g.hw_days.mean():5.1f} jours en canicule (>=3 j) | "
      f"{g.hot.mean():5.1f} jours Tmax>=40")
P("")
P("Annees avec le plus de jours de canicule :")
for y, v in annual.hw_days.sort_values(ascending=False).head(6).items():
    P(f"  {y}: {int(v)} jours")
P("")
P(f"Part des jours a z_tmax>=2 sur la reference : {df.loc[ref.index, 'heat_flag'].mean():.2%} "
  "(attendu ~2.3 % si loi normale ; Tmax est asymetrique a droite)")
e26 = df.loc["2026-06-01":"2026-08-31"]
P(f"Ete 2026 (juin-aout) : {int(e26.heat_flag.sum())} jours z>=2 sur {len(e26)}, "
  f"{int(e26.heatwave_flag.sum())} jours en canicule, {int(e26.heat_extreme_flag.sum())} jours z>=3")
P("")
both = ((df.heat_flag == 1) & (df.iforest_flag == 1)).sum()
P(f"Isolation Forest : {int(df.iforest_flag.sum())} jours signales (1 % de la reference) ; "
  f"{both} d'entre eux sont aussi des jours z_tmax>=2 ; "
  f"{int(((df.iforest_flag == 1) & (df.heat_flag == 0)).sum())} sont des anomalies que le z-score seul ne voit pas.")
top = df.loc[(df.iforest_flag == 1) & (df.heat_flag == 0)].sort_values("iforest_score", ascending=False).head(5)
P("  Exemples d'anomalies multivariees sans alerte chaleur :")
for d, r in top.iterrows():
    P(f"   {d:%Y-%m-%d}: Tmax {r.temperature_max_c:.1f} C, pluie {r.rain_sum_mm:.1f} mm, z_hr {r.z_rh:+.1f}, z_vent {r.z_wind:+.1f}")
(REPORTS_DIR / "validation_summary.txt").write_text("\n".join(lines), encoding="utf-8")
print("\n".join(lines))

# exports pour les figures
annual.to_csv(PROCESSED_DIR / "annual_summary.csv", float_format="%.3f")
