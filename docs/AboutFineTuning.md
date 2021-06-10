# About Fine Tuning

- 素直に学習してしまうと、ネットワークの最終層が過剰適合する危険性がある
  - lr をごとに調整することで対応できる？
- Fine-Tuning
  - BERT を新しいタスクを対象に再学習する
- Further Pre-training
  - BERT とは異なるデータで BERT と同じタスクを学習させてより新しいタスクに沿った重みを得る
- Multi-Task Fine-Tuning
- BERT は seq_len=512 で学習している

## https://arxiv.org/pdf/1905.05583.pdf

- 結論
  1. BERT の最上層はテキスト分類において有効
  2. 適切な層ごと学習率の設定により、BERT の重みを忘却させずに済む
  3. 事前学習により性能を大幅に向上
  4. 複数タスク追加学習は、単一の追加学習にも貢献するが、恩恵は小さくなる
  5. BERT は小さいデータサイズのタスクを改善させる

## Reference

- https://flyyufelix.github.io/2016/10/03/fine-tuning-in-keras-part1.html
- https://arxiv.org/pdf/1905.05583.pdf
