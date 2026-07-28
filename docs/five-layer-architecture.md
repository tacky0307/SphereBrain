# Sphere Brain Five-Layer Architecture v0.1

## 目的

Sphere Brainの内部を、外部表現から独立した数値活動として保つ。

```text
Encoder -> Core -> Trace -> Decoder
                    |
                    v
                Reflection -> Core
```

## Encoder

文字、音声、画像などをCoreへ直接渡さず、`NumericStimulus`へ変換する。
現在の `stable-hash-v1` は研究用の仮Encoderであり、意味理解を完成させるものではない。

## Core

数値ノード、接続、重み、伝播、経路強化だけを扱う。
Coreは文字列を受け取らず、外部表現を保存しない。

## Trace

実際に起きた活動を改変せず保存する。

- source_nodes
- activated_nodes
- traversed_edges
- activation_history
- final_activation
- learn_enabled

外部入力は `external_events` に分離し、Traceそのものを言語記憶にしない。
既存の `memories` テーブルは削除せず、そのまま残す。

## Decoder

Coreの最終状態を外部から観測可能な数値要約へ変換する。
最初のDecoderは文章を生成せず、上位ノード、活性数、経路数、活動ステップ、収束状態を返す。

## Reflection

Traceを文章へ戻さず、過去の数値活動から刺激ノードを作ってCoreへ再入力する。
Reflectionによる学習は `learn` フラグで明示的に切り替える。

## 研究用コマンド

```bash
python research_cycle.py experience "空は青い"
python research_cycle.py reflect
python research_cycle.py reflect --no-learn
```

## 現段階で意図的に行わないこと

- 日本語回答の生成
- Traceの意味づけ
- 外部文章の検索による回答
- 既存Web画面への即時統合

まず5層の境界が保たれることを確認し、その後にEncoderとDecoderの研究を進める。
