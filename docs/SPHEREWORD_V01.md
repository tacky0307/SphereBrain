# SphereWord v0.1

ローカルブラウザ (`127.0.0.1`) で遊ぶ、LLM → SphereBrain Core → LLM の言葉遊びゲーム。

## ゲーム

1. プレイヤーがお題を入力する（例: `海`）
2. LLMが8個の連想候補を生成する
3. 各候補を数値刺激へ変換し、本物の `SphereBrain` Coreへ非学習Probeする
4. 現在Coreに形成されているNode/Edge活動とEdge usageから秘密語を1つ選ぶ
5. プレイヤーは秘密語を推測する
6. 推測と秘密語のCore活動の重なりを距離として表示する
7. 秘密語・推測の経験がCoreへ入り、経路が少しずつ変化する

LLMは候補を生成するが秘密語は選ばない。Core外に状態→答えの表や正解ルールは持たない。

## 起動

```powershell
git switch experiment/sphereword-web-v0.1
git pull
python -m sphereword
```

または Windows で `run_sphereword.bat` をダブルクリック。

ブラウザは自動で次を開く。

```text
http://127.0.0.1:8765
```

終了はターミナルで `Ctrl+C`。

## 初回に必要なもの

既存の LLM → Core → LLM 実験と同様、OpenAI APIキーを環境変数で利用する。

```powershell
$env:OPENAI_API_KEY="..."
```

依存関係が未導入なら:

```powershell
pip install -r requirements.txt
```

## データ分離

SphereWord専用Coreは以下に保存される。

```text
data/sphereword_v01/brain.json
data/sphereword_v01/projection.npy
```

既存の `data/llm_core_v1/` や Semantic Encoder 系のCoreは変更しない。

## Coreの役割

候補 `c` について `お題 ｜ c` をEmbedding→固定ランダム射影→Core入口Nodeへ変換し、実Coreを `learn=False` で伝播させる。

候補順位は次のCore由来量だけで決める。

- お題活動とのNode overlap
- お題活動とのEdge overlap
- その候補Probeで通るEdgeの既存usage

秘密語決定後のみ、その経験を `learn=True` でCoreへ入れる。

推測の近さも、秘密語と推測のCore活動におけるNode/Edge overlapで計算する。

## モデル

既定値:

- Embedding: `text-embedding-3-small`
- Word candidate LLM: `gpt-5-mini`

環境変数で変更可能:

```text
SPHERE_EMBEDDING_MODEL
SPHERE_WORD_MODEL
```

## Coreリセット

SphereWord専用Coreだけを初期化して起動:

```powershell
python -m sphereword --reset-core
```

既存研究Coreは削除しない。
