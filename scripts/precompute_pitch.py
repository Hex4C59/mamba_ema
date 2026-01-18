"""离线预计算并缓存每条音频的 pitch 序列（与模型无关、可复用）。

功能
----
- 读取 CCSEMO 数据集的标签文件，按 split_set 列划分 train/val/test。
- 使用 Praat/Parselmouth 提取每条音频的原始 pitch（未归一化）。
- 将结果保存到缓存目录：data/processed/pitch_cache_ccsemo/<stem>.npy

参数
----
- --split: train|val|test|all，默认 all。
- --force: 存在同名缓存时是否强制重算并覆盖。

使用示例
--------
提取 CCSEMO 全部 split:
    uv run python scripts/precompute_pitch.py --split all

仅提取 train split:
    uv run python scripts/precompute_pitch.py --split train
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import tempfile
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import parselmouth
from parselmouth.praat import call
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class PitchFeatures:
    """从 single utterance 中提取原始 pitch 轨迹的辅助类.

    这里保留与原 audio_parsing_dataset.py 中相同的实现,
    确保离线提取与旧版在线提取逻辑一致。
    """

    def __init__(self, sound, temp_dir: str = "./tmp_pitch") -> None:
        self.sound = sound
        self.pitch_tiers = None
        self.total_duration = None
        self.pitch_point: list[float] = []
        self.time_point: list[float] = []

        self.pd: list[float] = []
        self.pt: list[float] = []
        self.ps: list[float | None] = []
        self.pr: list[float] = []

        os.makedirs(temp_dir, exist_ok=True)
        self.temp_dir = temp_dir
        self.temp_file = os.path.join(temp_dir, "PitchTier")

    def get_pitch_tiers(self):
        manipulation = call(self.sound, "To Manipulation", 0.01, 75, 600)
        self.pitch_tier = call(manipulation, "Extract pitch tier")
        return self.pitch_tier

    def stylize_pitch(self):
        if self.pitch_tier is not None:
            call(self.pitch_tier, "Stylize...", 2.0, "semitones")
            tmp_pitch_point = self.pitch_point
            tmp_time_point = self.time_point
            self.set_time_and_pitch_point()
            if len(self.pitch_point) == 0:
                self.pitch_point = tmp_pitch_point
                self.time_point = tmp_time_point
        else:
            print("pitch_tier is None")
            return

    def set_total_duration(self):
        total_duration_match = re.search(
            r"Total duration: (\d+(\.\d+)?) seconds",
            str(self.pitch_tier),
        )
        if total_duration_match:
            self.total_duration = float(total_duration_match.group(1))
        else:
            print("Total duration not found.")

    def set_time_and_pitch_point(self):
        self.pitch_tier.save(self.temp_file)
        r_file = open(self.temp_file, "r")

        self.pitch_point = []
        self.time_point = []
        while True:
            line = r_file.readline()
            if not line:
                break

            if "number" in line:
                value = re.sub(r"[^0-9^.]", "", line)

                if value.count(".") > 1:
                    parts = value.split(".")
                    value = parts[0] + "".join(parts[1:])
                if value != "":
                    self.time_point.append(round(float(value), 4))
            elif "value" in line:
                value = re.sub(r"[^0-9^.]", "", line)

                if value.count(".") > 1:
                    parts = value.split(".")
                    value = parts[0] + "".join(parts[1:])
                if value != "":
                    self.pitch_point.append(round(float(value), 4))

        if len(self.pitch_point) == 0:
            while True:
                line = r_file.readline()
                if not line:
                    break
        r_file.close()

    def get_pitchs(self):
        """返回 (pitch_values, time_points) 序列, 与旧实现保持一致."""
        self.pitch_tiers = self.get_pitch_tiers()
        self.set_time_and_pitch_point()
        return (self.pitch_point, self.time_point)


# ===== 配置参数 =====
# 标签文件路径
LABEL_CSV_PATH = PROJECT_ROOT / "data/label/CCSEMO/labels.csv"
# pitch 缓存目录
PITCH_CACHE_DIR = PROJECT_ROOT / "data/processed/pitch_cache_ccsemo"
# ===== 配置参数结束 =====


def load_label_splits(csv_path: str | Path) -> Dict[str, pd.DataFrame]:
    """从 CSV 文件加载数据并按 split_set 列划分。

    Returns:
        Dict with keys 'train', 'val', 'test', each containing a DataFrame.
    """
    df = pd.read_csv(csv_path)
    if "split_set" not in df.columns:
        raise ValueError("CSV must contain 'split_set' column")

    splits = {}
    for split in ["train", "val", "test"]:
        splits[split] = df[df["split_set"] == split].reset_index(drop=True)
    return splits


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="预计算 CCSEMO pitch 缓存")
    parser.add_argument(
        "--split",
        type=str,
        choices=["train", "val", "test", "all"],
        default="all",
        help="选择要处理的数据划分，默认 all",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="强制重新计算已存在的缓存文件",
    )
    return parser.parse_args()


def compute_pitch_array(wav_path: str, base_temp_dir: str) -> np.ndarray | None:
    """从单条 wav 提取 pitch 序列, 在独立临时目录中运行 Praat.

    为避免长期复用同一个临时文件导致的文件系统异常，这里为每条样本
    创建一个独立的子目录, 用后立即删除。
    """
    # 为当前样本创建独立临时目录
    os.makedirs(base_temp_dir, exist_ok=True)
    temp_dir = tempfile.mkdtemp(prefix="pitch_", dir=base_temp_dir)
    try:
        try:
            sound = parselmouth.Sound(wav_path)
        except Exception as e:  # noqa: BLE001
            print(f"[skip] Praat 打开失败: {wav_path}, err={e}")
            return None

        intonation = PitchFeatures(sound, temp_dir=temp_dir)
        pitch_tiers, time_points = intonation.get_pitchs()
        if not time_points or not pitch_tiers:
            # 提取不到有效 pitch 时，使用长度为 1 的全零向量占位，
            # 这样仍然会写入缓存并在训练阶段保留该样本。
            return np.zeros((1,), dtype=np.float32)
        return np.asarray(pitch_tiers, dtype=np.float32)
    finally:
        # 清理当前样本的临时目录, 避免堆积大量 PitchTier 文件
        try:
            shutil.rmtree(temp_dir)
        except Exception:
            pass


def run_one_split(
    split: str,
    df: pd.DataFrame,
    cache_root: Path,
    force: bool = False,
) -> Tuple[int, int]:
    """处理单个 split 的所有音频文件。

    Args:
        split: 数据划分名称 (train/val/test)
        df: 该 split 的 DataFrame
        cache_root: pitch 缓存目录
        force: 是否强制重新计算

    Returns:
        (成功缓存数, 总数)
    """
    done = 0
    total = len(df)

    # 每次调用各自的临时目录，避免多进程/多次调用冲突
    base_temp_dir = f"./tmp_pitch_precompute_{split}"
    os.makedirs(base_temp_dir, exist_ok=True)
    os.makedirs(cache_root, exist_ok=True)

    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"{split}"):
        name = row["name"]
        wav = row["audio_path"]

        # 生成缓存文件路径
        stem = os.path.splitext(os.path.basename(name))[0]
        out_path = cache_root / f"{stem}.npy"

        if (not force) and out_path.exists():
            done += 1
            continue

        pitch_arr = compute_pitch_array(wav, base_temp_dir=base_temp_dir)
        if pitch_arr is None:
            continue
        try:
            np.save(out_path, pitch_arr)
            done += 1
        except Exception as e:
            print(f"[warn] 无法保存: {out_path}, err={e}")

    return done, total


def main() -> None:
    cli = parse_args()

    print(f"[config] labels={LABEL_CSV_PATH}")
    print(f"[config] cache_dir={PITCH_CACHE_DIR}")

    # 加载数据并按 split_set 划分
    splits = load_label_splits(LABEL_CSV_PATH)

    # 确定要处理的 split
    todo = [cli.split] if cli.split != "all" else ["train", "val", "test"]

    total_done = 0
    total_all = 0
    for sp in todo:
        if sp not in splits or len(splits[sp]) == 0:
            print(f"[skip] split '{sp}' 为空或不存在")
            continue
        done, alln = run_one_split(sp, splits[sp], PITCH_CACHE_DIR, force=cli.force)
        print(f"[split {sp}] cached: {done}/{alln}")
        total_done += done
        total_all += alln

    print(f"[summary] cached: {total_done}/{total_all}")


if __name__ == "__main__":
    main()
