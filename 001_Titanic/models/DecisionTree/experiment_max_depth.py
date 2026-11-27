from pathlib import Path

import matplotlib

# 画面を開かず画像に保存する。VSCode以外の環境でも同じ処理を実行できる。
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree


# 1. 仮説：深さの制限により過学習が抑えられ、検証正解率が改善する。
# 比較対象以外の条件を変えると改善の理由が分からなくなるため、
# 特徴量・前処理・分割はExp001と同じにする。元データは読み込むだけ。
project_dir = Path(__file__).resolve().parents[2]
train = pd.read_csv(project_dir / "data" / "train.csv")
feature_columns = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare"]
# 特徴量は予測の入力、目的変数Survivedは予測したい正解（死亡0、生存1）。
X = train[feature_columns].copy()
y = train["Survived"]
# 文字列を数値に変換する。数字の大小が性別の優劣を意味するわけではない。
X["Sex"] = X["Sex"].map({"male": 0, "female": 1})
X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.2, random_state=42
)
# 未知の乗客を想定する検証用データの情報を学習に混ぜないよう、
# 欠損値を埋める中央値は学習用712件だけから算出する。
train_medians = X_train.median()
X_train = X_train.fillna(train_medians)
X_valid = X_valid.fillna(train_medians)

# 2. GridSearchCVを使わず、各条件を順に学習して比較する。
# Noneは深さ制限なし。モデルの他のパラメータはExp001と同じである。
max_depth_values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 20, None]
records = []
best_model = None
best_accuracy = -1.0
best_depth = None
for max_depth in max_depth_values:
    model = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
    # fitで分類ルールを学び、predictでそのルールを使った予測を得る。
    model.fit(X_train, y_train)
    # 正解率は、正しく予測した人数を評価した全人数で割った値。
    train_accuracy = accuracy_score(y_train, model.predict(X_train))
    valid_accuracy = accuracy_score(y_valid, model.predict(X_valid))
    records.append({
        "max_depth": "None" if max_depth is None else str(max_depth),
        "actual_depth": model.get_depth(),
        "node_count": model.tree_.node_count,
        "leaf_count": model.get_n_leaves(),
        "train_accuracy": train_accuracy,
        "validation_accuracy": valid_accuracy,
        "accuracy_gap": train_accuracy - valid_accuracy,
    })
    # 浅い設定から順に比較し、正解率が厳密に上がった時だけ更新する。
    # これにより同率なら浅い設定を選ぶ。Noneは最後に評価する。
    if valid_accuracy > best_accuracy:
        best_accuracy = valid_accuracy
        best_depth = max_depth
        best_model = model

results = pd.DataFrame(records)
print(f"学習用: {len(X_train)}件、検証用: {len(X_valid)}件")
print(results.to_string(index=False, float_format=lambda value: f"{value:.6f}"))
print(f"最良のmax_depth: {best_depth}、検証正解率: {best_accuracy:.2%}")
print("同じ検証データで条件を選んだ結果であり、未知データでの改善を確定するものではありません。")

# 3. 結果はローカルのanalysisに保存し、Git管理対象から除外する。
# 提出CSVや台帳、main.pyはこのスクリプトから書き換えない。
output_dir = project_dir / "analysis"
output_dir.mkdir(exist_ok=True)
results.to_csv(output_dir / "exp002_max_depth_results.csv", index=False)

# 横軸は比較条件を等間隔に並べたカテゴリ軸。Noneは無制限として区別する。
# 数値軸ではないので、10→12などの間隔も同じ幅になる。
positions = list(range(len(results)))
labels = ["None\n(unlimited)" if value == "None" else value for value in results["max_depth"]]
best_index = results["validation_accuracy"].idxmax()

fig, ax = plt.subplots(figsize=(11, 6))
ax.plot(positions, results["train_accuracy"], marker="o", label="Training")
ax.plot(positions, results["validation_accuracy"], marker="o", label="Validation")
# 最良値を点と注記で示す。同率の場合は最初の（浅い）条件を示す。
ax.scatter(best_index, best_accuracy, s=170, marker="*", color="red", zorder=5)
ax.annotate(f"Best: depth={best_depth}, accuracy={best_accuracy:.2%}",
            (best_index, best_accuracy), xytext=(15, -40), textcoords="offset points",
            arrowprops={"arrowstyle": "->"})
ax.set_xticks(positions, labels)
ax.set(xlabel="max_depth (categorical)", ylabel="Accuracy", title="Exp002: Depth and Accuracy")
ax.legend()
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(output_dir / "exp002_accuracy.png", dpi=180)
plt.close(fig)

# 正解率の差が大きいほど学習データへの合わせすぎが疑われる。
# ただし差が小さいだけでは良いモデルとは限らず、両正解率も確認する。
fig, ax = plt.subplots(figsize=(11, 6))
ax.plot(positions, results["accuracy_gap"], marker="o")
ax.axhline(0, color="gray", linewidth=1)
ax.set_xticks(positions, labels)
ax.set(xlabel="max_depth (categorical)", ylabel="Training accuracy - validation accuracy",
       title="Exp002: Depth and Accuracy Gap")
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(output_dir / "exp002_accuracy_gap.png", dpi=180)
plt.close(fig)

# ノード・葉の数は木の複雑さの目安。深さ制限を緩めた時の変化を見る。
fig, ax = plt.subplots(figsize=(11, 6))
ax.plot(positions, results["node_count"], marker="o", label="Nodes")
ax.plot(positions, results["leaf_count"], marker="o", label="Leaves")
ax.set_xticks(positions, labels)
ax.set(xlabel="max_depth (categorical)", ylabel="Count", title="Exp002: Model Complexity")
ax.legend()
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(output_dir / "exp002_complexity.png", dpi=180)
plt.close(fig)

# 最良モデルは学習用712件でfitした木のまま描く（全体で再学習しない）。
# 条件が真なら左、偽なら右。giniは分類の混ざり具合、samplesは人数、
# valueは[死亡数, 生存数]、classは多数派による予測結果を示す。
# 深い木が選ばれても全体を確認できるよう、大きな画像で保存する。
fig, ax = plt.subplots(figsize=(max(20, best_model.get_n_leaves() * 0.5),
                                max(10, best_model.get_depth() * 1.2)))
plot_tree(best_model, feature_names=feature_columns, class_names=["Died (0)", "Survived (1)"],
          filled=True, impurity=True, proportion=False, fontsize=7, precision=3, ax=ax)
ax.set_title(f"Exp002: Best Validation Model (max_depth={best_depth})")
fig.savefig(output_dir / "exp002_best_tree.png", dpi=180, bbox_inches="tight")
plt.close(fig)
print(f"結果CSV・画像の保存先: {output_dir}")
