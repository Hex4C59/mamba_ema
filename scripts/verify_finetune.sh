#!/bin/bash
# 快速验证解冻训练效果（fold 1，20 epochs）

echo "========================================"
echo "解冻 WavLM 微调验证"
echo "========================================"
echo ""
echo "配置:"
echo "  - Fold: 1"
echo "  - Epochs: 20 (快速验证)"
echo "  - Batch size: 4"
echo "  - Accumulation steps: 2 (等效 batch_size=8)"
echo "  - WavLM: 解冻 (lr=1e-5)"
echo "  - Other params: lr=1e-4"
echo ""
echo "预期:"
echo "  - 训练时间: ~2-3 小时"
echo "  - 显存占用: ~12-13 GB"
echo "  - CCC-V: 0.59 → 0.62-0.65"
echo ""

# 检查 GPU 显存
echo "检查 GPU 状态..."
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
echo ""

# 询问是否继续
read -p "是否开始训练? (y/n): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]
then
    echo "取消训练"
    exit 1
fi

# 开始训练
echo ""
echo "========================================"
echo "开始训练..."
echo "========================================"
echo ""

uv run python src/train.py \
    --config configs/finetune.yaml \
    --fold 1 \
    --gpu 0

echo ""
echo "========================================"
echo "训练完成！"
echo "========================================"
echo ""

# 查找最新的实验目录
LATEST_RUN=$(ls -td runs/ema_iemocap_finetune_*fold1* 2>/dev/null | head -1)

if [ -n "$LATEST_RUN" ]; then
    echo "实验结果: $LATEST_RUN"
    echo ""

    if [ -f "$LATEST_RUN/best_result.txt" ]; then
        echo "最佳结果:"
        cat "$LATEST_RUN/best_result.txt"
        echo ""

        # 提取 CCC-V 值
        CCC_V=$(grep "ccc_v:" "$LATEST_RUN/best_result.txt" | awk '{print $2}')
        echo "CCC-V: $CCC_V"

        # 与基线对比
        BASELINE_CCC_V=0.5896
        if [ -n "$CCC_V" ]; then
            IMPROVEMENT=$(python3 << EOF
baseline = $BASELINE_CCC_V
current = $CCC_V
delta = current - baseline
percent = (delta / baseline) * 100
print(f"提升: {delta:+.4f} ({percent:+.2f}%)")
if current >= 0.62:
    print("✓ 达到预期目标 (≥0.62)")
else:
    print("✗ 未达到预期目标 (需要 ≥0.62)")
EOF
)
            echo "$IMPROVEMENT"
        fi
    else
        echo "未找到 best_result.txt"
    fi
else
    echo "未找到实验结果目录"
fi

echo ""
echo "下一步:"
echo "  - 如果 CCC-V ≥ 0.62: 训练完整 50 epochs 并运行 5 fold"
echo "  - 如果 CCC-V < 0.62: 检查训练日志，调整超参数"
