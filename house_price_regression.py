import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ── 1. Load data ─────────────────────────────────────────────────────────────
df = pd.read_csv("train.csv")

FEATURES = ["GrLivArea", "BedroomAbvGr", "FullBath", "HalfBath"]
TARGET   = "SalePrice"

X = df[FEATURES].copy()
y = df[TARGET].copy()

print("=" * 55)
print("  House Price Prediction — Linear Regression")
print("=" * 55)
print(f"\nDataset: {len(df):,} houses | Features: {FEATURES}")

# ── 2. Train / test split ────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ── 3. Scale features ────────────────────────────────────────────────────────
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

# ── 4. Train model ───────────────────────────────────────────────────────────
model = LinearRegression()
model.fit(X_train_s, y_train)

# ── 5. Evaluate ──────────────────────────────────────────────────────────────
y_pred = model.predict(X_test_s)

mae  = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2   = r2_score(y_test, y_pred)

cv_r2 = cross_val_score(
    LinearRegression(), X_train_s, y_train,
    cv=5, scoring="r2"
)

print("\n── Model Coefficients ──────────────────────────────")
print(f"  Intercept : ${model.intercept_:>12,.0f}")
for feat, coef in zip(FEATURES, model.coef_):
    label = feat.ljust(14)
    direction = "▲" if coef > 0 else "▼"
    print(f"  {label}: ${coef:>10,.0f}  {direction} per std-dev")

print("\n── Test-Set Metrics ────────────────────────────────")
print(f"  R²   : {r2:.4f}  ({r2*100:.1f}% of variance explained)")
print(f"  MAE  : ${mae:>10,.0f}")
print(f"  RMSE : ${rmse:>10,.0f}")
print(f"\n── 5-Fold CV R² ────────────────────────────────────")
print(f"  Scores : {[f'{s:.4f}' for s in cv_r2]}")
print(f"  Mean   : {cv_r2.mean():.4f}  ±  {cv_r2.std():.4f}")

# ── 6. Sample predictions ─────────────────────────────────────────────────────
print("\n── Sample Predictions (first 8 test houses) ────────")
print(f"  {'Actual':>10}  {'Predicted':>10}  {'Error':>10}")
print("  " + "-" * 35)
for actual, pred in zip(y_test.values[:8], y_pred[:8]):
    err = pred - actual
    print(f"  ${actual:>9,.0f}  ${pred:>9,.0f}  {err:>+10,.0f}")

# ── 7. Visualise ─────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(15, 10))
fig.patch.set_facecolor("#0f172a")

gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.40, wspace=0.35)

ACCENT = "#38bdf8"
GREEN  = "#4ade80"
ORANGE = "#fb923c"
RED    = "#f87171"
TEXT   = "#e2e8f0"
PANEL  = "#1e293b"

def style_ax(ax, title):
    ax.set_facecolor(PANEL)
    ax.set_title(title, color=TEXT, fontsize=11, fontweight="bold", pad=10)
    ax.tick_params(colors=TEXT, labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor("#334155")
    ax.xaxis.label.set_color(TEXT)
    ax.yaxis.label.set_color(TEXT)

# 7a. Actual vs Predicted
ax1 = fig.add_subplot(gs[0, :2])
ax1.scatter(y_test / 1e3, y_pred / 1e3, alpha=0.45, s=20, color=ACCENT, edgecolors="none")
lims = [min(y_test.min(), y_pred.min()) / 1e3, max(y_test.max(), y_pred.max()) / 1e3]
ax1.plot(lims, lims, color=GREEN, lw=1.5, linestyle="--", label="Perfect fit")
ax1.set_xlabel("Actual Price ($k)")
ax1.set_ylabel("Predicted Price ($k)")
style_ax(ax1, "Actual vs Predicted Sale Price")
ax1.legend(facecolor=PANEL, edgecolor="#334155", labelcolor=TEXT, fontsize=8)
ax1.text(0.04, 0.92, f"R² = {r2:.3f}", transform=ax1.transAxes,
         color=GREEN, fontsize=10, fontweight="bold")

# 7b. Coefficients bar chart
ax2 = fig.add_subplot(gs[0, 2])
colors = [GREEN if c > 0 else RED for c in model.coef_]
labels = ["Sq Ft\n(GrLivArea)", "Bedrooms", "Full Bath", "Half Bath"]
bars = ax2.barh(labels, model.coef_, color=colors, edgecolor="none", height=0.55)
ax2.axvline(0, color="#475569", linewidth=0.8)
for bar, val in zip(bars, model.coef_):
    ax2.text(val + (500 if val >= 0 else -500), bar.get_y() + bar.get_height() / 2,
             f"${val:,.0f}", va="center",
             ha="left" if val >= 0 else "right", color=TEXT, fontsize=7.5)
style_ax(ax2, "Feature Coefficients\n($ per std-dev)")
ax2.set_xlabel("Coefficient ($)")

# 7c. Residuals vs Predicted
residuals = y_test.values - y_pred
ax3 = fig.add_subplot(gs[1, 0])
ax3.scatter(y_pred / 1e3, residuals / 1e3, alpha=0.45, s=18, color=ORANGE, edgecolors="none")
ax3.axhline(0, color=GREEN, lw=1.2, linestyle="--")
ax3.set_xlabel("Predicted Price ($k)")
ax3.set_ylabel("Residual ($k)")
style_ax(ax3, "Residuals vs Predicted")

# 7d. Residual distribution
ax4 = fig.add_subplot(gs[1, 1])
ax4.hist(residuals / 1e3, bins=35, color=ACCENT, edgecolor="#0f172a", linewidth=0.4, alpha=0.85)
ax4.axvline(0, color=GREEN, lw=1.2, linestyle="--")
ax4.set_xlabel("Residual ($k)")
ax4.set_ylabel("Count")
style_ax(ax4, "Residual Distribution")

# 7e. Metrics summary card
ax5 = fig.add_subplot(gs[1, 2])
ax5.set_facecolor(PANEL)
ax5.axis("off")
style_ax(ax5, "Model Performance")
metrics = [
    ("R²  (test)",    f"{r2:.4f}"),
    ("MAE",           f"${mae:,.0f}"),
    ("RMSE",          f"${rmse:,.0f}"),
    ("CV R² mean",    f"{cv_r2.mean():.4f}"),
    ("CV R² ± std",   f"± {cv_r2.std():.4f}"),
    ("Train samples", f"{len(X_train):,}"),
    ("Test samples",  f"{len(X_test):,}"),
]
for i, (label, value) in enumerate(metrics):
    y_pos = 0.85 - i * 0.12
    ax5.text(0.05, y_pos, label, transform=ax5.transAxes,
             color="#94a3b8", fontsize=9)
    ax5.text(0.95, y_pos, value, transform=ax5.transAxes,
             color=GREEN, fontsize=9, fontweight="bold", ha="right")

fig.suptitle("House Price Linear Regression — Feature Analysis",
             color=TEXT, fontsize=14, fontweight="bold", y=0.98)

out = "house_price_regression.png"
plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
print(f"\nChart saved → {out}")
plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.show()

print(f"\nChart saved → {out}")
