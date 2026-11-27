from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


# 1. CSVを読み込む。
# このファイル自身の場所を基準にすると、どのフォルダーから実行しても
# 同じ元データを読み込める。元のCSVには書き込まない。
model_dir = Path(__file__).resolve().parents[2]
data_dir = model_dir / "data"
train = pd.read_csv(data_dir / "train.csv")
test = pd.read_csv(data_dir / "test.csv")

# 2. 特徴量と目的変数を取り出す。
# 特徴量は予測の手がかりとなる入力、目的変数は予測したい正解である。
# Survivedは0が死亡、1が生存を表す。test.csvにはこの正解がない。
# PassengerIdは提出時の識別にだけ使い、モデルの入力には含めない。
feature_columns = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare"]
X = train[feature_columns].copy()
y = train["Survived"]
X_test = test[feature_columns].copy()

# 3. 性別を数値に変換する。
# scikit-learnの決定木には文字列をそのまま渡せないため、
# maleを0、femaleを1に置き換える。この数字は優劣を意味しない。
# 対応表は固定なので、訓練データと提出用データに同じ変換を使える。
sex_mapping = {"male": 0, "female": 1}
X["Sex"] = X["Sex"].map(sex_mapping)
X_test["Sex"] = X_test["Sex"].map(sex_mapping)

# 4・5. まず学習用80%、検証用20%に分割してから欠損値を補完する。
# 学習に使った乗客だけで評価すると、正解を覚えただけでも高得点になる。
# 未使用の検証用データで評価し、未知の乗客への予測性能を確認する。
# random_stateを固定することで、実行するたびに同じ分割を再現できる。
X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 欠損値は年齢などが記録されていない箇所である。
# 入力を数値でそろえるため、各列の中央値（順に並べた中央の値）で埋める。
# 中央値は極端に大きい値の影響を受けにくい代表値である。
# 検証用データの情報が学習側へ混ざる「データ漏洩」を防ぐため、
# 分割前の全データではなく、学習用データだけから中央値を求める。
# 検証用データにも、この学習用データの中央値をそのまま適用する。
train_medians = X_train.median()
X_train = X_train.fillna(train_medians)
X_valid = X_valid.fillna(train_medians)

# 6. 決定木を学習する。
# fitは特徴量と正解の組から、生存・死亡を分けるルールを学ぶ処理。
# Exp003の交差検証で選んだ最大深さ7を採用し、過度に細かい分岐を制限する。
# random_state=42で再現性を保ち、その他のパラメータはデフォルト値を使う。
model = DecisionTreeClassifier(max_depth=7, random_state=42)
model.fit(X_train, y_train)

# 7. 学習用・検証用それぞれの正解率を表示する。
# predictは、fitで学んだルールを使い、特徴量から予測値を返す処理。
# 予測する際に正解を渡す必要はない。正解は後で評価にだけ使う。
train_predictions = model.predict(X_train)
valid_predictions = model.predict(X_valid)

# 正解率 = 正しく予測した人数 / 評価した全人数。
# accuracy_scoreは予測と正解を1人ずつ比較して、この割合を計算する。
# 学習用だけ高く、検証用が低い場合は、学習データに合わせすぎた
# 「過学習」の可能性がある。ここでの単一分割評価はCVの評価値とは区別する。
train_accuracy = accuracy_score(y_train, train_predictions)
valid_accuracy = accuracy_score(y_valid, valid_predictions)
print(f"学習用データ: {len(X_train)}行、検証用データ: {len(X_valid)}行")
print(f"学習用の正解率: {train_accuracy:.4f} ({train_accuracy:.2%})")
print(f"検証用の正解率: {valid_accuracy:.4f} ({valid_accuracy:.2%})")

# 8. 提出用の予測に向けて、訓練データ全体で再学習する。
# 評価が終わったので、ここでは以前の検証用データも学習に使える。
# 中央値も訓練データ全体から計算し直す。test.csvからは計算しない。
full_train_medians = X.median()
X_full = X.fillna(full_train_medians)
X_test = X_test.fillna(full_train_medians)
model.fit(X_full, y)

# 9. 正解が与えられていないtest.csvの乗客について予測する。
test_predictions = model.predict(X_test)

# 10. Kaggle提出用CSVを作る。
# 必要な列はPassengerId（乗客の識別番号）とSurvived（0または1）の2列。
# test.csvと同じ順番・同じ人数で、各乗客の予測を1行ずつ記録する。
# index=Falseにより、pandasの行番号が余分な3列目として保存されるのを防ぐ。
submission = pd.DataFrame(
    {"PassengerId": test["PassengerId"], "Survived": test_predictions}
)
# Exp003の提出用として保存する。既存の提出ファイルは上書きしない。
submission_dir = model_dir / "submissions"
submission_dir.mkdir(exist_ok=True)
submission_path = submission_dir / "exp003.csv"
# mode="x"は新規作成専用。再実行で既存CSVを誤って上書きするのを防ぐ。
submission.to_csv(submission_path, index=False, mode="x")
print(f"提出用CSV: {submission_path}")
print(f"列名: {submission.columns.tolist()}、行数: {len(submission)}")
