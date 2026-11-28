from pathlib import Path
import runpy

import matplotlib

# ウィンドウを開かず、ローカルの画像ファイルへ描画する。
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.tree import DecisionTreeClassifier


# 1. Exp002と同じ特徴量・目的変数を用意する。
# 特徴量は予測の入力、Survivedは予測したい正解（死亡0、生存1）。
# データは読み取るだけで、提出CSVやmain.pyは変更しない。
project_dir = Path(__file__).resolve().parents[2]
train = pd.read_csv(project_dir / "data" / "train.csv")
feature_columns = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare"]
X = train[feature_columns].copy()
y = train["Survived"]
# 文字列は数値へ変換する。固定の対応表なので全体に適用しても、
# 検証データから統計量を学習するような情報漏洩は起こらない。
X["Sex"] = X["Sex"].map({"male": 0, "female": 1})

# 2. 全891件を5分割する。各回で4区画を学習、残り1区画を検証に使い、
# すべての乗客が検証用としてちょうど1回登場する。
# Stratifiedは、生存・死亡の割合を各区画で可能な限りそろえる指定。
# 分割を最初にリストへ保存し、全max_depthで同じ乗客の組を比較する。
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
splits = list(cv.split(X, y))
min_samples_leaf_values = [1, 2, 3, 4, 5, 7, 10, 15, 20, 30]
records = []
summaries = []
for min_samples_leaf in min_samples_leaf_values:
    leaf_label = str(min_samples_leaf)
    fold_records = []
    for fold, (train_indices, valid_indices) in enumerate(splits, start=1):
        # ilocは行番号で抽出する操作。学習用と検証用は重ならない。
        X_train = X.iloc[train_indices].copy()
        X_valid = X.iloc[valid_indices].copy()
        y_train = y.iloc[train_indices]
        y_valid = y.iloc[valid_indices]
        # 欠損値補完はFoldごとにやり直す。中央値を全891件から求めると、
        # 検証用の情報が学習側に漏れるため、当該Foldの学習用のみで計算する。
        medians = X_train.median()
        X_train = X_train.fillna(medians)
        X_valid = X_valid.fillna(medians)

        # 毎回モデルを新規生成し、前のFoldで学んだルールを持ち越さない。
        # 深さ以外のパラメータはExp002と同じ。fitで学習、predictで予測する。
        model = DecisionTreeClassifier(
            max_depth=7,
            min_samples_leaf=min_samples_leaf,
            random_state=42
        )
        model.fit(X_train, y_train)
        record = {
            "min_samples_leaf": leaf_label,
            "fold": fold,
            "train_count": len(X_train),
            "validation_count": len(X_valid),
            # 正解率は、正しく予測した人数÷評価した全人数である。
            "train_accuracy": accuracy_score(y_train, model.predict(X_train)),
            "validation_accuracy": accuracy_score(y_valid, model.predict(X_valid)),
            "actual_depth": model.get_depth(),
            "node_count": model.tree_.node_count,
            "leaf_count": model.get_n_leaves(),
        }
        records.append(record)
        fold_records.append(record)

    # 3. 各深さの5回の結果を集計する。各Foldを同じ重みで平均する。
    # 標準偏差は分割による揺れの目安。指定どおりddof=0を使う。
    # これは信頼区間でも、独立した未知データでの性能保証でもない。
    folds = pd.DataFrame(fold_records)
    mean_train = folds["train_accuracy"].mean()
    mean_valid = folds["validation_accuracy"].mean()
    summaries.append({
        "min_samples_leaf": leaf_label,
        "mean_train_accuracy": mean_train,
        "mean_validation_accuracy": mean_valid,
        "std_validation_accuracy": folds["validation_accuracy"].std(ddof=0),
        "min_validation_accuracy": folds["validation_accuracy"].min(),
        "max_validation_accuracy": folds["validation_accuracy"].max(),
        "mean_accuracy_gap": mean_train - mean_valid,
        "mean_node_count": folds["node_count"].mean(),
        "mean_leaf_count": folds["leaf_count"].mean(),
    })

details = pd.DataFrame(records)
summary = pd.DataFrame(summaries)
# 浅い条件から並べているので、idxmaxが最初の最大値を返すことにより、
# 平均検証正解率が同率なら浅い設定を優先する。Noneは最後に評価する。
best_index = summary["mean_validation_accuracy"].idxmax()
best = summary.loc[best_index]
print("Foldごとの計測結果:")
print(details.to_string(index=False, float_format=lambda value: f"{value:.6f}"))
print("min_samples_leafごとの集計結果（標準偏差ddof=0）:")
print(summary.to_string(index=False, float_format=lambda value: f"{value:.6f}"))
print(f"\n最良min_samples_leaf: {best['min_samples_leaf']}、平均検証正解率: {best['mean_validation_accuracy']:.2%}")
print("これは条件選択用のCV評価であり、独立した未知データでの確定的な性能ではありません。")

output_dir = project_dir / "analysis"
output_dir.mkdir(exist_ok=True)
details.to_csv(output_dir / "exp004_leaf_details.csv", index=False)
summary.to_csv(output_dir / "exp004_leaf_summary.csv", index=False)

# 4. グラフの横軸は等間隔のカテゴリ。Noneは無制限として区別する。
# ±標準偏差の帯は分割によるばらつきを示すもので、信頼区間ではない。
positions = list(range(len(summary)))
labels = summary["min_samples_leaf"].tolist()
fig, ax = plt.subplots(figsize=(11, 6))
ax.plot(positions, summary["mean_train_accuracy"], marker="o", label="Mean training")
ax.plot(positions, summary["mean_validation_accuracy"], marker="o", label="Mean validation")
ax.fill_between(positions,
                summary["mean_validation_accuracy"] - summary["std_validation_accuracy"],
                summary["mean_validation_accuracy"] + summary["std_validation_accuracy"],
                alpha=0.2, label="Validation +/- 1 SD (ddof=0)")
ax.scatter(best_index, best["mean_validation_accuracy"], marker="*", s=180, color="red", zorder=5)
ax.annotate(f"Best: leaf={best['min_samples_leaf']}, {best['mean_validation_accuracy']:.2%}",
            (best_index, best["mean_validation_accuracy"]), xytext=(15, -40),
            textcoords="offset points", arrowprops={"arrowstyle": "->"})
ax.set_xticks(positions, labels)
ax.set(xlabel="min_samples_leaf", ylabel="Accuracy", title="Exp004: Stratified 5-Fold CV")
ax.legend()
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(output_dir / "exp004_leaf_accuracy.png", dpi=180)
plt.close(fig)

fig, ax = plt.subplots(figsize=(11, 6))
for fold in range(1, 6):
    values = details.loc[details["fold"] == fold, "validation_accuracy"]
    ax.plot(positions, values, marker="o", label=f"Fold {fold}")
ax.set_xticks(positions, labels)
ax.set(xlabel="min_samples_leaf", ylabel="Validation accuracy", title="Exp004: Each Fold")
ax.legend()
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(output_dir / "exp004_leaf_folds.png", dpi=180)
plt.close(fig)

# 箱は四分位範囲、中央線は中央値。点で全5件を重ね、標本数の少なさも示す。
fig, ax = plt.subplots(figsize=(11, 6))
distributions = [details.loc[details["min_samples_leaf"] == value, "validation_accuracy"].to_numpy()
                 for value in summary["min_samples_leaf"]]
ax.boxplot(distributions, positions=positions, widths=0.5, showfliers=False)
for position, values in zip(positions, distributions):
    # 点が重ならないよう横へ少しずらす。正解率の値そのものは変えない。
    ax.scatter([position + offset for offset in [-0.16, -0.08, 0, 0.08, 0.16]], values,
               s=22, color="tab:blue", alpha=0.8, zorder=3)
ax.set_xticks(positions, labels)
ax.set(xlabel="min_samples_leaf", ylabel="Validation accuracy", title="Exp004: Five Observations per Depth")
ax.grid(axis="y", alpha=0.3)
fig.tight_layout()
fig.savefig(output_dir / "exp004_leaf_boxplot.png", dpi=180)
plt.close(fig)

