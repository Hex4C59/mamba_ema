#!/bin/bash
# 并行训练 5 fold（快速验证，20 epochs）

echo "========================================"
echo "并行训练 5 Fold - 快速验证（20 epochs）"
echo "========================================"
echo ""
echo "配置:"
echo "  - Epochs: 20"
echo "  - Batch size: 8"
echo "  - WavLM: 全层解冻 (lr=1e-5)"
echo "  - 并行: 5 GPU (0-4)"
echo ""
echo "预期:"
echo "  - 总时间: ~2-2.5 小时"
echo "  - 显存/卡: ~16-18 GB"
echo "  - CCC-V: 0.59 → 0.64-0.68"
echo ""

# 检查 GPU 可用性
echo "检查 GPU 状态..."
nvidia-smi --query-gpu=index,name,memory.free --format=csv
echo ""

# 询问是否继续
read -p "是否开始训练 5 fold? (y/n): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]
then
    echo "取消训练"
    exit 1
fi

# 创建日志目录
mkdir -p logs/finetune_parallel

# 启动 5 个 fold 并行训练
echo ""
echo "========================================"
echo "启动并行训练..."
echo "========================================"
echo ""

CUDA_VISIBLE_DEVICES=0 uv run python src/train.py \
    --config configs/finetune.yaml --fold 1 --gpu 0 \
    > logs/finetune_parallel/fold1.log 2>&1 &
PID1=$!
echo "Fold 1 启动 (PID: $PID1, GPU: 0)"

CUDA_VISIBLE_DEVICES=1 uv run python src/train.py \
    --config configs/finetune.yaml --fold 2 --gpu 0 \
    > logs/finetune_parallel/fold2.log 2>&1 &
PID2=$!
echo "Fold 2 启动 (PID: $PID2, GPU: 1)"

CUDA_VISIBLE_DEVICES=2 uv run python src/train.py \
    --config configs/finetune.yaml --fold 3 --gpu 0 \
    > logs/finetune_parallel/fold3.log 2>&1 &
PID3=$!
echo "Fold 3 启动 (PID: $PID3, GPU: 2)"

CUDA_VISIBLE_DEVICES=3 uv run python src/train.py \
    --config configs/finetune.yaml --fold 4 --gpu 0 \
    > logs/finetune_parallel/fold4.log 2>&1 &
PID4=$!
echo "Fold 4 启动 (PID: $PID4, GPU: 3)"

CUDA_VISIBLE_DEVICES=4 uv run python src/train.py \
    --config configs/finetune.yaml --fold 5 --gpu 0 \
    > logs/finetune_parallel/fold5.log 2>&1 &
PID5=$!
echo "Fold 5 启动 (PID: $PID5, GPU: 4)"

echo ""
echo "所有 fold 已启动！"
echo ""
echo "监控命令:"
echo "  - 查看所有日志: tail -f logs/finetune_parallel/*.log"
echo "  - 查看 Fold 1: tail -f logs/finetune_parallel/fold1.log"
echo "  - 查看 GPU 使用: watch -n 1 nvidia-smi"
echo ""

# 等待所有任务完成
echo "等待训练完成..."
wait $PID1 $PID2 $PID3 $PID4 $PID5

echo ""
echo "========================================"
echo "所有 fold 训练完成！"
echo "========================================"
echo ""

# 收集结果
echo "收集结果..."
echo ""

for fold in {1..5}; do
    LATEST_RUN=$(ls -td runs/ema_iemocap_finetune_iemocap_fold${fold}_* 2>/dev/null | head -1)

    if [ -n "$LATEST_RUN" ]; then
        if [ -f "$LATEST_RUN/best_result.txt" ]; then
            echo "=== Fold $fold ==="
            cat "$LATEST_RUN/best_result.txt"
            echo ""
        fi
    fi
done

# 计算平均值
echo "========================================"
echo "5 Fold 平均结果"
echo "========================================"

python3 << 'EOF'
import os
import re
from pathlib import Path

results = []
for fold in range(1, 6):
    # 找到最新的实验目录
    pattern = f"runs/ema_iemocap_finetune_iemocap_fold{fold}_*"
    dirs = sorted(Path(".").glob(pattern), key=os.path.getmtime, reverse=True)

    if dirs:
        result_file = dirs[0] / "best_result.txt"
        if result_file.exists():
            content = result_file.read_text()
            ccc_v = float(re.search(r'ccc_v: ([\d.]+)', content).group(1))
            ccc_a = float(re.search(r'ccc_a: ([\d.]+)', content).group(1))
            ccc_avg = float(re.search(r'ccc_avg: ([\d.]+)', content).group(1))
            results.append({'fold': fold, 'ccc_v': ccc_v, 'ccc_a': ccc_a, 'ccc_avg': ccc_avg})

if results:
    avg_v = sum(r['ccc_v'] for r in results) / len(results)
    avg_a = sum(r['ccc_a'] for r in results) / len(results)
    avg_avg = sum(r['ccc_avg'] for r in results) / len(results)

    print(f"CCC-V:   {avg_v:.4f}")
    print(f"CCC-A:   {avg_a:.4f}")
    print(f"CCC-Avg: {avg_avg:.4f}")
    print()

    # 与基线对比
    baseline_v = 0.5896
    baseline_a = 0.6867
    baseline_avg = 0.6381

    delta_v = avg_v - baseline_v
    delta_a = avg_a - baseline_a
    delta_avg = avg_avg - baseline_avg

    print(f"提升 (vs v1 基线):")
    print(f"  CCC-V:   {delta_v:+.4f} ({delta_v/baseline_v*100:+.2f}%)")
    print(f"  CCC-A:   {delta_a:+.4f} ({delta_a/baseline_a*100:+.2f}%)")
    print(f"  CCC-Avg: {delta_avg:+.4f} ({delta_avg/baseline_avg*100:+.2f}%)")
    print()

    if avg_v >= 0.64:
        print("✓ 达到预期目标 (CCC-V ≥ 0.64)")
    else:
        print(f"✗ 未达到预期目标 (CCC-V {avg_v:.4f} < 0.64)")
else:
    print("未找到结果文件")
EOF

echo ""
echo "下一步:"
echo "  - 如果效果好：运行完整 50 epochs 训练"
echo "  - 更新实验记录: runs/experiment_comparison.md"
