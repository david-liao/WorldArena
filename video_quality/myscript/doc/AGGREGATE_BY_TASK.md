# aggregate_by_task.py 使用说明

按 RoboTwin 2.0 任务类别，对 `aggregated_results.csv` 里的每个指标求均值。

脚本：`video_quality/myscript/aggregate_by_task.py`

---

## 功能

1. 从聚合结果 CSV 的 `Video_ID` 字段（形如 `fixed_scene_task_episode123`）提取 `episodeK`。
2. 左连接 episode → task 映射（默认用 `classify_test_dataset_tasks.py` 产出的 `task_classification_output/episode_task_mapping.csv`）。
3. 对所有数值列按 `predicted_task` 求均值（自动忽略空值），并输出每个任务的样本数 `n_episodes`。
4. 追加一行 `__ALL__` 作为整体均值。
5. 输出 CSV，可选输出 Markdown 表格。

---

## 依赖

- Python 3（项目环境自带）
- `pandas`（已随 WorldArena 环境安装）

无需额外依赖；Markdown 输出使用脚本内置的简易 writer，不需要 `tabulate`。

---

## 输入 / 输出

### 输入
- `results_csv`（位置参数，必填）：聚合结果 CSV，必须包含 `Video_ID` 列，以及任意数量的数值指标列。
- `--mapping PATH`（可选）：episode → task 映射 CSV，默认：
  `video_quality/myscript/task_classification_output/episode_task_mapping.csv`
  只要求包含 `episode_id`、`predicted_task` 两列。

### 输出
- CSV：默认写到 `<results_csv 同目录>/per_task_means.csv`，可用 `-o` 指定。
- TSV：传 `--tsv` 时写出，默认 `<results_csv 同目录>/per_task_means.tsv`，方便直接粘贴进 Excel / 飞书表格（制表符分隔，数值 6 位小数）。也可 `--tsv PATH` 指定路径。
- Markdown：仅在传 `--markdown PATH` 时写出。
- 终端：完整打印整张表（4 位小数格式）。

输出列：`predicted_task, n_episodes, <每个数值指标的均值…>`，末行 `__ALL__` 为整体均值。

---

## 命令行参数

| 参数 | 说明 |
| --- | --- |
| `results_csv` | 聚合结果 CSV 路径（位置参数） |
| `--mapping PATH` | 指定自定义 episode→task 映射 CSV |
| `-o, --output PATH` | 输出 CSV 路径 |
| `--markdown PATH` | 额外输出 Markdown 表 |
| `--tsv [PATH]` | 额外输出 TSV（制表符分隔）；不带值时写到默认路径 `per_task_means.tsv` |
| `--sort COLUMN` | 排序列，默认 `predicted_task`；可传任意指标列名 |
| `--ascending` | 升序排序（默认按任务名升序、按指标降序） |

---

## 常用示例

### 1. 最小用法

```bash
cd /mydir/code/WorldArena
python video_quality/myscript/aggregate_by_task.py \
    video_quality/csv_results/wan/aggregated_results.csv
```

产物：`video_quality/csv_results/wan/per_task_means.csv`

### 2. 同时输出 Markdown

```bash
python video_quality/myscript/aggregate_by_task.py \
    video_quality/csv_results/wan/aggregated_results.csv \
    --markdown video_quality/csv_results/wan/per_task_means.md
```

### 2b. 同时输出 TSV（贴 Excel 用）

```bash
# 默认写到 <results_csv 同目录>/per_task_means.tsv
python video_quality/myscript/aggregate_by_task.py \
    video_quality/csv_results/wan/aggregated_results.csv --tsv

# 或指定路径
python video_quality/myscript/aggregate_by_task.py \
    video_quality/csv_results/wan/aggregated_results.csv \
    --tsv video_quality/csv_results/wan/per_task_means.tsv
```

在 Excel 里直接 `Ctrl+A` → `Ctrl+C` → 粘贴到 Excel 即可自动按列分开。

### 3. 按某个指标降序排序（快速看谁最好 / 最差）

```bash
python video_quality/myscript/aggregate_by_task.py \
    video_quality/csv_results/wan/aggregated_results.csv \
    --sort "Subject Consistency"
```

### 4. 批量处理多个模型

```bash
for model in wan vace step-645000-ema_infer; do
    python video_quality/myscript/aggregate_by_task.py \
        "video_quality/csv_results/${model}/aggregated_results.csv" \
        --markdown "video_quality/csv_results/${model}/per_task_means.md"
done
```

### 5. 使用自定义分类映射

```bash
python video_quality/myscript/aggregate_by_task.py \
    video_quality/csv_results/wan/aggregated_results.csv \
    --mapping path/to/my_episode_task_mapping.csv \
    -o path/to/my_per_task_means.csv
```

---

## 行为细节

- `Video_ID` 里找不到 `episodeK` 的行会给出 warning 但不会中断。
- 连接不上映射表的 episode 会被丢弃统计，并给出 warning（终端打印数量）。
- 如果指标列整列为空（例如当前 `wan` 的 `Semantic Alignment / Depth Accuracy / Trajectory Accuracy / JEPA Similarity`），对应均值会显示为 `NaN`/空白，列不会被删除；补上数据后直接重跑即可。
- `n_episodes` 是该任务下的样本数（通常为 20）；若输入 CSV 缺失某些 episode，这个值会相应变小。
- 默认映射覆盖 `fixed_scene_task` 的 1000 条 episode，其它来源的 Video_ID（例如 `val_dataset` 或其它前缀）如需统计需另行提供 `--mapping`。
