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

現在のmain.pyはExp003提出準備用です。max_depth=7を採用し、全891件で再学習した予測を`submissions/exp003.csv`へ新規作成します。既存ファイルがある場合はエラーで停止し、上書きしません。exp001.csvなどの過去のCSVも保持します。

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

## Exp002：最大深さによる過学習抑制

```powershell
cd D:\working\kaggle\001_Titanic
.\.venv\Scripts\python.exe models/DecisionTree/experiment_max_depth.py
```

max_depthを1〜10、12、15、20、Noneで比較します。その他の条件はExp001と同一です。実際の深さ・ノード数・葉数・両正解率・差をコンソールとanalysis/exp002_max_depth_results.csvへ出力します。正解率、正解率の差、複雑さ、最良モデルの木をexp002_accuracy.png、exp002_accuracy_gap.png、exp002_complexity.png、exp002_best_tree.pngとして同じanalysisに保存します。出力はGit管理対象外です。

2026-11-28の結果ではmax_depth=5が最良で、学習正解率85.81%、検証正解率80.45%、差5.37ポイントでした。実際の深さ5、47ノード、24葉です。Exp001の97.89%・75.98%と比較して、検証正解率は4.47ポイント改善しました。制限なしの条件はExp001の結果を再現しています。

この分割では深さ制限による過学習抑制の仮説を支持します。ただし、同じ179件の検証データを使って14条件から選んだ結果で、選択による楽観的な評価の可能性があります。独立したテストや交差検証なしに汎化性能の改善を断言しません。Kaggle未提出で、main.pyへの自動適用も行いません。同率の場合は浅い設定を選びます。

実験の仮説・最良max_depth・要約はexperiments.csvのparametersおよびnotesへ記録し、git_commitは実装と結果説明のコミットを指します。既存のExp001行は保持します。


## Exp003：Stratified 5-Fold CVによる深さの選択

```powershell
cd D:\working\kaggle\001_Titanic
.\.venv\Scripts\python.exe models/DecisionTree/experiment_cross_validation.py
```

親実験は002です。全891件をStratifiedKFold(n_splits=5, shuffle=True, random_state=42)で分割し、生存・死亡の割合を各Foldで可能な限りそろえます。分割を一度だけ作成し、Exp002と同じ14種類のmax_depthで共有します。各Foldでは新しいモデルを生成し、学習用だけの中央値で欠損値を補完します。全員が検証用に1回ずつ登場します。処理はforループで明示しています。

analysisにexp003_cv_details.csv（70行）、exp003_cv_summary.csv（14行）、exp003_cv_accuracy.png、exp003_cv_folds.png、exp003_cv_boxplot.png、exp003_comparison.pngを出力します。標準偏差はddof=0、平均は5個のFold正解率の単純平均です。同率なら浅い設定を選びます。Exp002の結果CSVがなければ既存のexperiment_max_depth.pyを実行して再現します。生成ファイルはGit管理対象外です。

2026-11-28の最良はmax_depth=7、平均学習88.24%、平均検証82.71%、検証の標準偏差1.36ポイントでした。max_depth=5は平均検証81.82%、標準偏差1.60ポイント、Noneは79.35%、標準偏差3.72ポイントです。5はNoneより平均検証で2.47ポイント高く、ばらつきも小さくなりました。3〜9付近の深さ制限は比較的良い結果ですが、5が常に最良という仮説までは支持しません。

Exp002のmax_depth=5は単一分割で80.45%、今回の同条件は81.82%です。分割方法が異なるため、この差をモデル改善とは解釈しません。今回の最良条件も同じCV結果から選んでいるため、独立した未知データでの確定的な性能ではありません。帯は平均±標準偏差で、信頼区間ではありません。Kaggle提出、main.pyへの適用、mainへのマージは行いません。

## Exp003の提出準備

CVで選んだDecisionTreeClassifier(max_depth=7, random_state=42)をmain.pyへ適用しました。特徴量・性別変換・欠損値補完・その他のパラメータは維持しています。提出時はtrain.csv全891件の中央値で補完し、全891件で再学習してtest.csvの418件を予測します。

submissions/exp003.csvはPassengerId、Survivedの2列です。418行、欠損なし、乗客IDの順序一致、予測が0/1のみであることを検証済みです。生存予測143人、死亡予測275人でした。提出CSVはGit管理対象外です。Kaggleへのアップロードはユーザーが手動で行います。

状態はready_to_submit、公開スコアは未記入です。CV平均学習88.2433%、平均検証82.7148%と標準偏差は従来のまま保持します。main.pyの単一分割評価（学習88.48%、検証76.54%）はCV評価と異なるため、台帳のCV結果へ上書きしません。タグtitanic-exp003は提出用モデルのコミットを指し、そのSHAは後続の台帳更新コミットで記録します。冒頭のベースライン説明と結果はExp001の履歴です。
