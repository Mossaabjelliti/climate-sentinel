from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT / "data" / "processed"
FIGURES_DIR = ROOT / "reports" / "figures"
D = pd.read_csv(PROCESSED_DIR/"climate_anomalies.csv", parse_dates=["date"], index_col="date")
S = pd.read_csv(PROCESSED_DIR/"spi_monthly.csv", parse_dates=[0], index_col=0)
A = pd.read_csv(PROCESSED_DIR/"annual_summary.csv", index_col=0)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
DRY, WET, INK, MUTE = "#C8612B", "#2B6CB0", "#14302B", "#6B756F"
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": MUTE, "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
                     "figure.dpi": 150, "savefig.bbox": "tight"})

# Fig 1 : Tmax 2026 vs normale +/- 2 sigma
y = D.loc["2026-01-01":"2026-09-01"]
sd = (y.temperature_max_c - y.tmax_normal) / y.z_tmax
fig, ax = plt.subplots(figsize=(9, 4.2))
ax.fill_between(y.index, y.tmax_normal-2*sd, y.tmax_normal+2*sd, color="#DDD6C4", label="Normale 1991-2020 ± 2 écarts-types")
ax.plot(y.index, y.tmax_normal, color=MUTE, lw=1.2, label="Normale du jour")
ax.plot(y.index, y.temperature_max_c, color=INK, lw=1, label="Tmax 2026")
h = y[y.heat_flag == 1]
ax.scatter(h.index, h.temperature_max_c, s=18, color=DRY, zorder=3, label="Anomalie chaude (z ≥ 2)")
ax.set_ylabel("Température maximale (°C)"); ax.set_title("Tataouine 2026 : température maximale et anomalies chaudes", loc="left", fontsize=12, color=INK)
ax.legend(frameon=False, fontsize=9, loc="upper left"); fig.savefig(FIGURES_DIR/"fig1_tmax_2026.png"); plt.close(fig)

# Fig 2 : SPI-12
s = S["spi12"].dropna()
fig, ax = plt.subplots(figsize=(9, 3.8))
ax.bar(s.index, s.clip(lower=-3, upper=3), width=31, color=np.where(s < 0, DRY, WET))
for v, lab in [(-1, "modérée"), (-1.5, "sévère"), (-2, "extrême")]:
    ax.axhline(v, color=MUTE, lw=0.7, ls="--"); ax.text(s.index[-1], v, f" {lab}", va="center", fontsize=8, color=MUTE)
ax.axhline(0, color=INK, lw=0.8)
for lbl, d in [("2000-01", "2001-10-01"), ("2021-22", "2021-12-01"), ("1981", "1981-12-01"), ("2013", "2013-09-01")]:
    ax.annotate(lbl, (pd.Timestamp(d), -2.4), ha="center", fontsize=9, color=INK)
ax.set_ylabel("SPI-12"); ax.set_ylim(-3, 3)
ax.set_title("Indice de précipitation standardisé sur 12 mois (orange : sec, bleu : humide)", loc="left", fontsize=12, color=INK)
fig.savefig(FIGURES_DIR/"fig2_spi12.png"); plt.close(fig)

# Fig 3 : pluie annuelle et jours chauds
fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.8))
normal = A.loc[1991:2020, "rain"].mean()
a1.bar(A.index, A.rain, color=np.where(A.rain < normal*0.5, DRY, "#9DB8C9"))
a1.axhline(normal, color=INK, lw=1); a1.text(1980, 235, f"Trait : normale 1991-2020 ({normal:.0f} mm)", fontsize=9, color=INK)
a1.set_title("Pluie annuelle (mm)", loc="left", fontsize=12, color=INK)
a2.bar(A.index, A.heat_days, color="#E3B9A0")
z = np.polyfit(A.index, A.heat_days, 1); a2.plot(A.index, np.polyval(z, A.index), color=DRY, lw=2)
a2.set_title("Jours d'anomalie chaude (z ≥ 2) par an", loc="left", fontsize=12, color=INK)
fig.tight_layout(); fig.savefig(FIGURES_DIR/"fig3_annuel.png"); plt.close(fig)
print("trend heat days/decade", round(z[0]*10, 2))
