Markdown
# LLM as a Policy Teacher for StarCraft II (Bachelor's Thesis Project)

## Overview
本リポジトリは、卒業研究にて大規模言語モデル（LLM）の戦略知識を深層強化学習（RL）エージェントへ知識蒸留する手法を、リアルタイムストラテジーゲーム『StarCraft II』へ適用・拡張した実験用コードです。

先行研究である [LLM4Teach](https://arxiv.org/abs/2311.13373) のフレームワークをベースに、StarCraft II の複雑な状態空間を LLM が解釈可能なテキスト表現に変換するモジュールや、LLM の出力を PPO 方策に組み込む中間表現アダプターを独自に実装しています。

## Key Features
- **Environment Support**: StarCraft II (PySC2) への新規対応
- **State-to-Text Module**: 資源量、ユニット数、建物情報などの内部状態を自然言語へ変換
- **Knowledge Distillation**: LLM の戦略的出力を教師信号として PPO エージェントを学習
- **Efficiency**: 推論時間を LLM 直結時の数秒から数ミリ秒単位へ短縮

## Requirements
- Python 3.x
- PyTorch
- PySC2 (StarCraft II Learning Environment)
- OpenAI API Key (LLM Teacher として利用)

## Running Experiments

### Setup
環境変数に OpenAI API キーを設定してください。
```bash
export OPENAI_API_KEY='your-api-key-here'
```

### Train
以下のコマンドで、StarCraft II 環境における学習を開始します。
```bash
python main.py train --task starcraft2 --savedir train
```

### Evaluate
学習済みモデルの評価を行います。
```bash
python main.py eval --task starcraft2 --loaddir train --savedir eval
```

## Repository Structure
本リポジトリは実験用のため、主にロジックの構造確認を目的としています。
 - models/: PPO およびアダプターの実装
 - envs/: StarCraft II 接続および状態テキスト変換ロジック
 - main.py: 学習・評価のエントリポイント

## Acknowledgements
本研究の実装にあたり、LLM4Teach の構成を参考にさせていただきました。
