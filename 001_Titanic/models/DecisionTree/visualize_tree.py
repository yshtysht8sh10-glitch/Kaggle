from pathlib import Path

import matplotlib

# 画像ファイルとして保存するため、画面表示を必要としない描画方式を使う。
# VSCodeや画面のない環境でも、同じようにPNGを生成できる。
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree


# 1. main.pyと同じ訓練データ・特徴量を使う。
# この分析では検証評価時点の木を調べるため、test.csvへの予測や
# 訓練データ全体での再学習は行わない。提出済みCSVにも書き込まない。
# ファイル自身の場所を基準にし、実行時のカレントディレクトリに依存しない。
project_dir = Path(__file__).resolve().parents[2]
train = pd.read_csv(project_dir / "data" / "train.csv")
feature_columns = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare"]
# 特徴量は予測の手がかりとなる入力、目的変数は予測したい正解である。
X = train[feature_columns].copy()
y = train["Survived"]

# 2. 文字列をモデルが扱える数値にする。対応表はmain.pyと同じ。
X["Sex"] = X["Sex"].map({"male": 0, "female": 1})

# 3. main.pyと同じ分割にし、学習に使わない乗客で性能を確認する。
# random_state=42により、学習用712件と検証用179件を再現する。
X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.2, random_state=42
)
# 欠損値は列の中央値で補完する。検証用の情報が学習に混ざらないよう、
# 中央値は学習用データのみで求め、検証用にも同じ値を適用する。
train_medians = X_train.median()
X_train = X_train.fillna(train_medians)
X_valid = X_valid.fillna(train_medians)

# 4. fitで分類ルールを学ぶ。モデルの深さなどは制限せず、
# random_state以外はすべてデフォルトにして初回モデルを再現する。
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

# predictで予測し、正しく予測した人数÷全人数で正解率を計算する。
train_accuracy = accuracy_score(y_train, model.predict(X_train))
valid_accuracy = accuracy_score(y_valid, model.predict(X_valid))
print(f"学習用: {len(X_train)}件、検証用: {len(X_valid)}件")
# 深さは根から最も遠い葉までの分岐回数。根の深さは0である。
# ノード総数には分岐ノードと、最終的な分類を返す葉の両方を含む。
print(f"決定木の最大深さ: {model.get_depth()}")
print(f"ノード総数: {model.tree_.node_count}")
print(f"葉ノード数: {model.get_n_leaves()}")
print("特徴量重要度:")
# 重要度は各特徴量の分岐による不純度の減少を集計・正規化した値。
# 合計は1になる。因果関係や未知データでの有用性を直接示すものではなく、
# この学習データに対する木の使い方を理解するための目安である。
for feature, importance in zip(feature_columns, model.feature_importances_):
    print(f"  {feature}: {importance:.6f} ({importance:.2%})")
print(f"学習正解率: {train_accuracy:.4f} ({train_accuracy:.2%})")
print(f"検証正解率: {valid_accuracy:.4f} ({valid_accuracy:.2%})")

# 5. 分析画像の出力先を作る。analysisは.gitignoreでGit管理から除外する。
output_dir = project_dir / "analysis"
output_dir.mkdir(exist_ok=True)

# 図中は英語表記にして、日本語フォントの有無による文字化けを避ける。
# 各ノードの条件が真なら左、偽なら右へ進む。
# giniは生存・死亡の混ざり具合で、0なら全員が同じ分類である。
# samplesはそのノードに到達した学習用の人数、valueは[死亡数, 生存数]。
# classは多数派による予測結果で、色の濃さは分類の偏りを表す。
class_names = ["Died (0)", "Survived (1)"]

# 上位3階層は「根を1階層目」とし、深さ0・1・2を表示する。
# plot_treeのmax_depthは描画範囲だけの指定であり、モデルを変更しない。
# (...)は表示範囲より下にも木が続くことを示す省略記号である。
fig, ax = plt.subplots(figsize=(18, 10))
plot_tree(
    model, max_depth=2, feature_names=feature_columns, class_names=class_names,
    filled=True, impurity=True, proportion=False, precision=3, fontsize=10, ax=ax,
)
ax.set_title("Exp001: Top 3 Levels (root = level 1; display depth <= 2)")
top_path = output_dir / "exp001_tree_top3.png"
fig.savefig(top_path, dpi=180, bbox_inches="tight")
plt.close(fig)

# 木全体はノードが多いため、大きな画像として保存する。
# 全体の形を見る図なので、個々のラベルは画像を拡大して確認する。
fig, ax = plt.subplots(figsize=(60, 30))
plot_tree(
    model, feature_names=feature_columns, class_names=class_names,
    filled=True, impurity=True, proportion=False, precision=3, fontsize=5, ax=ax,
)
ax.set_title("Exp001: Full Validation-stage Tree (trained on 712 passengers)")
full_path = output_dir / "exp001_tree_full.png"
fig.savefig(full_path, dpi=200, bbox_inches="tight")
plt.close(fig)

# 6. 重要度を降順に並べ、どの入力が分岐に使われたか比較しやすくする。
importance = pd.Series(model.feature_importances_, index=feature_columns)
importance = importance.sort_values(ascending=True)
fig, ax = plt.subplots(figsize=(9, 5))
bars = ax.barh(importance.index, importance.values)
ax.bar_label(bars, fmt="%.4f", padding=4)
ax.set_xlim(0, importance.max() * 1.2)
ax.set_xlabel("Feature importance (normalized Gini decrease)")
ax.set_title("Exp001: Feature Importance")
fig.tight_layout()
importance_path = output_dir / "exp001_feature_importance.png"
fig.savefig(importance_path, dpi=180, bbox_inches="tight")
plt.close(fig)

for path in [top_path, full_path, importance_path]:
    print(f"画像保存先: {path}")
