"""Draw model architecture diagram."""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch


def draw_architecture():
    fig, ax = plt.subplots(1, 1, figsize=(14, 16))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 16)
    ax.axis("off")
    ax.set_aspect("equal")

    # Colors
    C_INPUT = "#E3F2FD"      # Light blue - inputs
    C_ENCODER = "#FFF3E0"    # Light orange - encoders
    C_FUSION = "#E8F5E9"     # Light green - fusion modules
    C_MAMBA = "#FCE4EC"      # Light pink - mamba
    C_HEAD = "#F3E5F5"       # Light purple - output head
    C_BORDER = "#424242"

    def add_box(x, y, w, h, text, color, fontsize=10, bold=False):
        box = FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.15",
            facecolor=color, edgecolor=C_BORDER, linewidth=1.5
        )
        ax.add_patch(box)
        weight = "bold" if bold else "normal"
        ax.text(x + w/2, y + h/2, text, ha="center", va="center",
                fontsize=fontsize, fontweight=weight, wrap=True)
        return x + w/2, y + h/2  # center

    def add_arrow(start, end, color="#666666", style="->"):
        arrow = FancyArrowPatch(
            start, end, arrowstyle=style, color=color,
            connectionstyle="arc3,rad=0", mutation_scale=12, linewidth=1.5
        )
        ax.add_patch(arrow)

    # Title
    ax.text(7, 15.5, "MultimodalEmotionModel Architecture (Offline Mode)",
            ha="center", va="center", fontsize=14, fontweight="bold")

    # ========== INPUT LAYER ==========
    y_input = 13.5
    # WavLM multi-layer features
    cx1, cy1 = add_box(0.5, y_input, 3.5, 1.2, "WavLM Features\n[B, L=4, T, 1024]", C_INPUT, 9)
    # X-Vector
    cx2, cy2 = add_box(5.2, y_input, 3.5, 1.2, "X-Vector\n[B, 512]", C_INPUT, 9)
    # Pitch
    cx3, cy3 = add_box(10, y_input, 3.5, 1.2, "Pitch (F0)\n[B, T]", C_INPUT, 9)

    # ========== LAYER FUSION ==========
    y_fusion = 11.8
    cx_lf, cy_lf = add_box(0.5, y_fusion, 3.5, 1.0, "LearnableLayerFusion\n(Softmax Weighted Sum)", C_FUSION, 9)
    add_arrow((cx1, y_input), (cx_lf, y_fusion + 1.0))

    # ========== ENCODERS ==========
    y_enc = 10.0
    # Speech Encoder
    cx_se, cy_se = add_box(0.5, y_enc, 3.5, 1.2, "OfflineSpeechEncoder\n(LayerNorm + Dropout)\n[B, T, 1024]", C_ENCODER, 9)
    add_arrow((cx_lf, y_fusion), (cx_se, y_enc + 1.2))

    # Speaker Encoder
    cx_spk, cy_spk = add_box(5.2, y_enc, 3.5, 1.2, "OfflineSpeakerEncoder\n(L2 Normalize)\n[B, 512]", C_ENCODER, 9)
    add_arrow((cx2, y_input), (cx_spk, y_enc + 1.2))

    # Pitch bypass (直接到后面)
    ax.annotate("", xy=(11.75, 4.5), xytext=(11.75, y_input),
                arrowprops=dict(arrowstyle="->", color="#666666", lw=1.5,
                               connectionstyle="arc3,rad=0"))

    # ========== FiLM ==========
    y_film = 8.0
    cx_film, cy_film = add_box(2.5, y_film, 5.0, 1.2, "FiLM (Feature-wise Linear Modulation)\nγ · h + β  (Speaker-conditioned)\n[B, T, 1024]", C_FUSION, 9)
    add_arrow((cx_se, y_enc), (4, y_film + 1.2))
    add_arrow((cx_spk, y_enc), (5.5, y_film + 1.2))

    # FiLM details box
    ax.text(8.5, 8.5, "γ, β = MLP(speaker)\nγ = tanh(γ) + 1", fontsize=8,
            ha="left", va="center", style="italic", color="#666")

    # ========== MAMBA ==========
    y_mamba = 5.8
    cx_mamba, cy_mamba = add_box(1.5, y_mamba, 7.0, 1.5, "MambaUpdater\n(2× Mamba Blocks with Residual)\nInput: [B, T, 1024] → Output: [B, T, 128]", C_MAMBA, 9, bold=True)
    add_arrow((cx_film, y_film), (cx_mamba, y_mamba + 1.5))

    # Mamba details
    mamba_detail = """d_model=256, d_state=16
d_conv=4, expand=2
Pre-norm + Residual"""
    ax.text(9.5, 6.5, mamba_detail, fontsize=8, ha="left", va="center",
            style="italic", color="#666", family="monospace")

    # ========== POOLING ==========
    y_pool = 4.0
    cx_pool, cy_pool = add_box(1.5, y_pool, 5.0, 1.0, "Masked Mean Pooling\n[B, T, 128] → [B, 128]", C_FUSION, 9)
    add_arrow((cx_mamba, y_mamba), (3.5, y_pool + 1.0))

    # ========== CONCAT ==========
    y_cat = 2.5
    cx_cat, cy_cat = add_box(3, y_cat, 6.0, 1.0, "Concatenate\n[B, 128] ⊕ [B, T] → [B, 128+T]", C_FUSION, 9)
    add_arrow((cx_pool, y_pool), (5, y_cat + 1.0))
    add_arrow((11.75, 4.5), (8, y_cat + 0.5))

    # ========== REGRESSION HEAD ==========
    y_head = 0.8
    cx_head, cy_head = add_box(2, y_head, 8.0, 1.2, "Regression Head\nLinear(→256) → LN → ReLU → Dropout\n→ Linear(→128) → LN → ReLU → Dropout → Linear(→2)", C_HEAD, 9)
    add_arrow((cx_cat, y_cat), (cx_head, y_head + 1.2))

    # ========== OUTPUT ==========
    ax.text(7, 0.3, "Output: Valence, Arousal [B, 2]", ha="center", va="center",
            fontsize=11, fontweight="bold", color="#1565C0")

    # ========== LEGEND ==========
    legend_y = 15.0
    legend_items = [
        (C_INPUT, "Input Features"),
        (C_ENCODER, "Encoders"),
        (C_FUSION, "Fusion Modules"),
        (C_MAMBA, "Mamba SSM"),
        (C_HEAD, "Output Head"),
    ]
    for i, (color, label) in enumerate(legend_items):
        x = 0.5 + i * 2.6
        box = FancyBboxPatch((x, legend_y), 0.4, 0.3, boxstyle="round,pad=0.02",
                             facecolor=color, edgecolor=C_BORDER, linewidth=1)
        ax.add_patch(box)
        ax.text(x + 0.5, legend_y + 0.15, label, fontsize=8, va="center")

    plt.tight_layout()
    plt.savefig("docs/model_architecture.png", dpi=150, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.savefig("docs/model_architecture.pdf", bbox_inches="tight",
                facecolor="white", edgecolor="none")
    print("Saved to docs/model_architecture.png and docs/model_architecture.pdf")


if __name__ == "__main__":
    draw_architecture()
