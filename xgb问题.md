# XGBoost 基线问题清单（代码正确性）

## 结论

`baseline_xgb/train_xgb.py` 当前可以运行，但在“未来预测（forecasting）”语义下，代码逻辑存在阻断问题：
- 目前是同刻目标拟合，不是未来时点预测。
- 在当前目标定义下，部分 `number` 衍生特征会形成同刻信息泄漏。

---

## 1) 阻断问题（Blocking）

### 1.1 目标定义未前移，缺少预测 horizon

- 文件：`baseline_xgb/train_xgb.py`
- 证据：
  - `TARGET_COL = "number"`
  - `model.fit(train_df[FEATURE_COLS], train_df[TARGET_COL], ...)`
- 问题：目标是同一行 `number`，没有 `t+H` 标签构造（无 `shift(-H)`）。
- 影响：训练语义偏向“同刻回归”而非“未来预测”。

### 1.2 同刻目标泄漏风险（在 forecasting 语义下）

- 文件：`data_process_and_data_to_use/build_features.py`
- 证据：多个训练特征直接使用当前行 `number` 参与计算，例如：
  - `feat_number_diff_1row = number.diff(1)`
  - `feat_number_pct_change_1row = (number - number.shift(1)) / ...`
  - `feat_number_momentum_* = number - number.shift(*)`
  - `feat_number_ma_* = number.rolling(...).mean()`（rolling 默认含当前行）
  - `feat_number_pos_48row = (number - rolling_min_48) / ...`
- 问题：若标签仍是同一行 `number`，这些特征会携带标签同刻信息。
- 影响：评估指标可能虚高，不能代表真实未来预测能力。

---

## 2) 非阻断但需明确的问题（Non-blocking）

### 2.1 训练切分正确但不能替代标签对齐

- 文件：`baseline_xgb/train_xgb.py`
- 现状：按时间排序后使用 chronological split（70/10/20）是正确的。
- 风险：仅靠正确切分无法修复“目标未前移”问题。

### 2.2 结构化特征可得性边界需与预测时点一致

- 文件：`baseline_xgb/train_xgb.py`（含 weather/holiday/traffic 特征）
- 风险：若任务是分钟级日内预测，部分日聚合天气特征可能在预测时点不可得。
- 说明：这是任务定义与特征可得性边界问题，不是脚本执行错误。

### 2.3 静态类型告警

- 文件：`baseline_xgb/train_xgb.py`
- 现状：存在 Pyright 告警（pandas 类型推断相关）。
- 影响：不影响当前运行，但影响静态质量门禁与维护稳定性。

---

## 3) 训练代码修正方向（最小改造）

1. 显式定义 forecast horizon `H`（例如 1、4、16 行）。
2. 构造标签：`y = number.shift(-H)`，并按标签可用范围裁剪尾部样本。
3. 重新检查所有输入特征在预测原点 `t` 的可得性，只允许使用 `<= t` 可见信息。
4. 保持时间顺序切分（已具备），并在报告中注明 `H` 与样本对齐规则。

---

## 4) 解释边界

如果明确把任务定义为“同刻估计（nowcasting）”，当前实现可以作为回归器使用；
但若定义为“未来预测（forecasting）”，则必须先完成上述标签前移与可得性对齐。
