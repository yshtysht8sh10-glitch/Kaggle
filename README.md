# Kaggle 学習記録

Python・pandas・scikit-learnを使い、データの前処理、モデル学習、検証、提出までを学びます。予測精度に加えて、読みやすい日本語コメントと再現可能な実験を重視します。

## 参加コンペ

| コンペ | 内容 | 初回モデル | 公開スコア |
|---|---|---|---:|
| [Titanic](https://www.kaggle.com/competitions/titanic) | 生存予測 | DecisionTreeClassifier | 0.72488 |

実行方法と結果は[001_Titanic/README.md](001_Titanic/README.md)を参照してください。

## 構成

```text
kaggle/
├── .gitignore
├── README.md
└── 001_Titanic/
    ├── data/                 # ローカルのみ
    ├── .venv/                # ローカルのみ
    ├── submissions/          # ローカルのみ
    │   └── exp001.csv        # 提出済み・上書き禁止
    ├── main.py
    ├── experiments.csv
    ├── requirements.txt
    └── README.md
```

## Gitによる実験管理

Gitのルートはこのkaggleフォルダーです。ソースコードの実験別コピーは作らず、コミット・ブランチ・タグで履歴を保存します。

- メインブランチ：`main`
- 実験ブランチ：`titanic/exp002-max-depth`など
- 実験タグ：`titanic-exp001`など
- コミットメッセージ：`Titanic: Add decision tree baseline`など

新しい実験はmainからブランチを作り、対象コンペのmain.pyを変更します。動作確認後にモデルのコミットを作成し、そのSHAをexperiments.csvのgit_commitへ記録します。モデルのコミットにタグを付け、台帳の更新は別コミットにします。評価結果・変更内容・改良元IDも台帳へ記録し、確認後にmainへ統合します。タグは付け替えず、履歴を書き換える強制Pushは行いません。

```powershell
git switch main
git switch -c titanic/exp002-max-depth
# main.pyを変更し、実行・評価する
git add 001_Titanic/main.py 001_Titanic/README.md
git commit -m "Titanic: Limit decision tree depth"
git rev-parse HEAD
git tag titanic-exp002
# 表示されたSHAとタグ、評価結果などをexperiments.csvへ記録する
git add 001_Titanic/experiments.csv
git commit -m "Titanic: Record exp002 results"
git switch main
git merge titanic/exp002-max-depth
git push origin main
git push origin titanic-exp002
```

データ・仮想環境・提出CSV・モデルバイナリ・認証ファイルはルートの.gitignoreで全コンペ共通に除外します。認証トークンをコードや実験台帳へ書かないでください。コミット前に`git diff --cached --name-only`と`git diff --cached`で公開内容を確認します。ローカル限定ファイルはGitHubから復元できないため、必要に応じて別途保管してください。

現時点では共通処理のモジュール化、MLflow、実験管理の自動化は導入しません。
