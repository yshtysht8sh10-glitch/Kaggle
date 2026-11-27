# Titanic：決定木ベースライン

## Exp001の決定木を分析する

既存の仮想環境にrequirements.txtの依存ライブラリをインストールし、次を実行します。

```powershell
cd D:\working\kaggle\001_Titanic
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe models/DecisionTree/visualize_tree.py
```

main.pyと同じ分割・前処理・モデル設定で、学習用712件から検証評価時点の木を再現します。訓練データ全体での再学習は行いません。深さ・ノード数・葉数・特徴量重要度・学習と検証の正解率を表示し、analysisフォルダーに以下を保存します。

- exp001_tree_top3.png：根を1階層目とする上位3階層（深さ0〜2）。描画範囲のみ制限し、モデルの深さは変更しません。
- exp001_tree_full.png：木全体。ラベルは画像を拡大して確認します。
- exp001_feature_importance.png：不純度の減少に基づく特徴量重要度。

木の条件が真なら左、偽なら右へ進みます。giniは分類の混ざり具合（0なら同一分類のみ）、samplesはそのノードの学習件数、valueは[死亡数, 生存数]、classは予測結果です。Sexの0は男性、1は女性です。重要度は因果関係を表すものではありません。分析画像はGit管理対象外で、再実行時は更新されます。main.py・実験台帳・提出CSVは変更しません。

## 目的・モデル

乗客情報から生存（1）・死亡（0）を予測します。読みやすさを優先した初回実験001です。

- モデル：`DecisionTreeClassifier(random_state=42)`。その他はデフォルト。
- 特徴量：Pclass、Sex、Age、SibSp、Parch、Fare。
- 目的変数：Survived。
- 前処理：Sexをmale=0、female=1へ変換し、欠損値を各列の中央値で補完。
- 検証：80%・20%に分割（random_state=42）。中央値は学習用データのみから計算し、検証用にも適用。
- 提出用予測：訓練データ全体で中央値を再計算し、全体で再学習。test.csvには訓練データ全体の中央値を適用。

特徴量エンジニアリング、パラメータ最適化、アンサンブルは行いません。

## 環境構築・実行

Python 3.12.10、pandas 2.2.3、scikit-learn 1.6.1で動作確認しています。GitHubにはデータと仮想環境が含まれないため、初回はTitanicコンペからtrain.csv、test.csv、gender_submission.csvを取得してdataフォルダーへ置きます。

```powershell
cd D:\working\kaggle\001_Titanic
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe models/DecisionTree/main.py
```

既存の.venvを使用する場合は環境作成を省略できます。仮想環境の有効化は不要です。入力・出力パスはmain.pyの位置から解決するため、カレントディレクトリに依存しません。

VSCodeではkaggleフォルダーを開き、Python拡張機能の「Python: Select Interpreter」で`001_Titanic\.venv\Scripts\python.exe`を選びます。`001_Titanic/models/DecisionTree/main.py`を開いて「Run Python File in Terminal」を実行します。

通常実行の出力は`submissions/submission.csv`です。再実行ではこの作業用ファイルが更新されます。提出済みの`submissions/exp001.csv`は保存用で、main.pyからは書き込みません。

## 初回実験の結果

| 項目 | 結果 |
|---|---:|
| 実験日 | 2026-11-26 |
| 学習用データ | 712行 |
| 検証用データ | 179行 |
| 学習正解率 | 0.9789（97.89%） |
| 検証正解率 | 0.7598（75.98%） |
| Kaggle公開スコア | 0.72488 |
| 提出CSV | 418行、PassengerIdとSurvivedの2列 |
| 状態 | submitted |
| タグ | titanic-exp001 |

Kaggleスコアは提出済みの結果としてユーザーから提供された値です。学習用と検証用の正解率に21.91ポイントの差があり、過学習の傾向が見られます。

構成変更後も正解率を再現し、生成予測とexp001.csvが一致すること、元データ3ファイルが変更されていないことを確認しています。モデル処理と日本語コメントを維持し、入出力パスのみ変更しました。

## 提出方法

KaggleのTitanicコンペ画面の「Submit Predictions」からCSVをアップロードします。提出CSVはPassengerId、Survivedの2列で、test.csvと同じ順序の418人です。今回の環境整備では再提出しません。

今後は作業用submission.csvを確認してから、新しい実験IDの名前（例：submissions/exp002.csv）でコピーして保管します。既存のexp001.csvなどは上書きしません。提出後は公開スコアと状態をexperiments.csvへ記録します。

## 実験記録

experiments.csvは1実験1行、UTF-8です。既存の12項目にgit_commit（モデルのコミットSHA）とgit_tagを追加しています。IDは文字列として扱い、先頭のゼロを維持します。parent_experimentは初回では空欄、改良時は元のIDを記録します。

モデルを先にコミットし、そのSHAを台帳へ記録して別コミットにするため、git_commitは台帳更新コミットではなくモデル本体のコミットを指します。初回のタグはtitanic-exp001です。モデル・実行結果・台帳は同じタグを基準に対応付けます。タグ時点のgit_commit欄は未記入で、後続の台帳更新コミットで確定します。

今後は実験別フォルダーを作らず、ブランチ上でmain.pyを変更します。具体的な命名規則と手順は[ルートREADME](../README.md)を参照してください。過去のモデルはコミット・タグで保存し、提出結果はローカルのsubmissionsに保管します。data、.venv、submissionsはGit管理対象外です。

## モデル別の構成

異なるモデルはmodels配下のフォルダーで分け、同一モデルの改善は仮説単位のGitブランチで管理します。

```text
001_Titanic/
├── data/
├── .venv/
├── submissions/
├── analysis/
├── models/
│   └── DecisionTree/
│       ├── main.py
│       └── visualize_tree.py
├── experiments.csv
├── requirements.txt
└── README.md
```

DecisionTreeの入出力はコンペルートを基準に解決します。VSCodeでは既存の001_Titanic/.venvを選択します。
