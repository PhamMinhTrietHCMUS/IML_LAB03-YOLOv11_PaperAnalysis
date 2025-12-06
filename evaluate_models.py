"""
Script to evaluate and compare YOLOv8, YOLOv9, and YOLOv11 models on test set
"""
import os
from pathlib import Path
from ultralytics import YOLO
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Set up paths
BASE_DIR = Path(r"D:\Yolo\NMMH_LAB3")
TEST_DIR = BASE_DIR / "rock-paper-scissors.v1i.yolov11" / "test"
RUNS_DIR = BASE_DIR / "runs" / "detect"

# Model paths
models = {
    "YOLOv8": RUNS_DIR / "yolov8" / "weights" / "best.pt",
    "YOLOv9": RUNS_DIR / "yolov9" / "weights" / "best.pt",
    "YOLOv11": RUNS_DIR / "yolov11" / "weights" / "best.pt"
}

# Create output directory for evaluation results
OUTPUT_DIR = BASE_DIR / "evaluation_results"
OUTPUT_DIR.mkdir(exist_ok=True)

print("=" * 70)
print("ĐÁNH GIÁ VÀ SO SÁNH CÁC MÔ HÌNH YOLO")
print("=" * 70)

# Store results
results_dict = {}
detailed_metrics = []

# Evaluate each model
for model_name, model_path in models.items():
    print(f"\n{'='*70}")
    print(f"Đang đánh giá mô hình: {model_name}")
    print(f"Model path: {model_path}")
    print(f"{'='*70}")
    
    if not model_path.exists():
        print(f"⚠ Không tìm thấy file weights: {model_path}")
        continue
    
    # Load model
    model = YOLO(str(model_path))
    
    # Validate on test set
    results = model.val(
        data=str(BASE_DIR / "data.yaml"),
        split="test",
        save_json=True,
        save_hybrid=False,
        conf=0.25,
        iou=0.45,
        project=str(OUTPUT_DIR),
        name=model_name,
        exist_ok=True
    )
    
    # Extract metrics
    metrics = {
        "Model": model_name,
        "Precision (P)": results.box.p.mean() if hasattr(results.box, 'p') else results.box.mp,
        "Recall (R)": results.box.r.mean() if hasattr(results.box, 'r') else results.box.mr,
        "mAP@50": results.box.map50,
        "mAP@50-95": results.box.map,
        "F1-Score": 2 * (results.box.mp * results.box.mr) / (results.box.mp + results.box.mr + 1e-10)
    }
    
    results_dict[model_name] = metrics
    detailed_metrics.append(metrics)
    
    # Print results
    print(f"\n📊 Kết quả đánh giá {model_name}:")
    print(f"  - Precision:    {metrics['Precision (P)']:.4f}")
    print(f"  - Recall:       {metrics['Recall (R)']:.4f}")
    print(f"  - F1-Score:     {metrics['F1-Score']:.4f}")
    print(f"  - mAP@50:       {metrics['mAP@50']:.4f}")
    print(f"  - mAP@50-95:    {metrics['mAP@50-95']:.4f}")

# Create comparison DataFrame
df_results = pd.DataFrame(detailed_metrics)
df_results = df_results.set_index("Model")

# Save to CSV
csv_path = OUTPUT_DIR / "comparison_results.csv"
df_results.to_csv(csv_path)
print(f"\n✅ Đã lưu kết quả so sánh vào: {csv_path}")

# Print comparison table
print("\n" + "=" * 70)
print("BẢNG SO SÁNH CÁC MÔ HÌNH")
print("=" * 70)
print(df_results.to_string())
print("=" * 70)

# Find best model for each metric
print("\n🏆 MÔ HÌNH TỐT NHẤT THEO TỪNG METRIC:")
for metric in df_results.columns:
    best_model = df_results[metric].idxmax()
    best_value = df_results[metric].max()
    print(f"  - {metric}: {best_model} ({best_value:.4f})")

# ================= VISUALIZATION =================
print("\n📈 Đang tạo biểu đồ so sánh...")

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (16, 10)
plt.rcParams['font.size'] = 10

# Create figure with subplots
fig = plt.figure(figsize=(18, 12))
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

# Color palette
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
model_names = df_results.index.tolist()

# 1. Bar chart comparison - All metrics
ax1 = fig.add_subplot(gs[0, :])
x = np.arange(len(df_results.columns))
width = 0.25
for i, model in enumerate(model_names):
    values = df_results.loc[model].values
    ax1.bar(x + i * width, values, width, label=model, color=colors[i], alpha=0.8)

ax1.set_xlabel('Metrics', fontsize=12, fontweight='bold')
ax1.set_ylabel('Score', fontsize=12, fontweight='bold')
ax1.set_title('So Sánh Tổng Quan Các Metrics', fontsize=14, fontweight='bold', pad=20)
ax1.set_xticks(x + width)
ax1.set_xticklabels(df_results.columns, rotation=15, ha='right')
ax1.legend(loc='upper right')
ax1.grid(axis='y', alpha=0.3)
ax1.set_ylim([0, 1.1])

# Add value labels on bars
for i, model in enumerate(model_names):
    values = df_results.loc[model].values
    for j, v in enumerate(values):
        ax1.text(j + i * width, v + 0.02, f'{v:.3f}', 
                ha='center', va='bottom', fontsize=8, fontweight='bold')

# 2. Precision comparison
ax2 = fig.add_subplot(gs[1, 0])
precision_values = df_results['Precision (P)'].values
bars = ax2.barh(model_names, precision_values, color=colors, alpha=0.8)
ax2.set_xlabel('Score', fontsize=11, fontweight='bold')
ax2.set_title('Precision', fontsize=12, fontweight='bold')
ax2.set_xlim([0, 1.0])
for i, (bar, val) in enumerate(zip(bars, precision_values)):
    ax2.text(val + 0.02, i, f'{val:.4f}', va='center', fontweight='bold')
ax2.grid(axis='x', alpha=0.3)

# 3. Recall comparison
ax3 = fig.add_subplot(gs[1, 1])
recall_values = df_results['Recall (R)'].values
bars = ax3.barh(model_names, recall_values, color=colors, alpha=0.8)
ax3.set_xlabel('Score', fontsize=11, fontweight='bold')
ax3.set_title('Recall', fontsize=12, fontweight='bold')
ax3.set_xlim([0, 1.0])
for i, (bar, val) in enumerate(zip(bars, recall_values)):
    ax3.text(val + 0.02, i, f'{val:.4f}', va='center', fontweight='bold')
ax3.grid(axis='x', alpha=0.3)

# 4. F1-Score comparison
ax4 = fig.add_subplot(gs[1, 2])
f1_values = df_results['F1-Score'].values
bars = ax4.barh(model_names, f1_values, color=colors, alpha=0.8)
ax4.set_xlabel('Score', fontsize=11, fontweight='bold')
ax4.set_title('F1-Score', fontsize=12, fontweight='bold')
ax4.set_xlim([0, 1.0])
for i, (bar, val) in enumerate(zip(bars, f1_values)):
    ax4.text(val + 0.02, i, f'{val:.4f}', va='center', fontweight='bold')
ax4.grid(axis='x', alpha=0.3)

# 5. mAP@50 comparison
ax5 = fig.add_subplot(gs[2, 0])
map50_values = df_results['mAP@50'].values
bars = ax5.barh(model_names, map50_values, color=colors, alpha=0.8)
ax5.set_xlabel('Score', fontsize=11, fontweight='bold')
ax5.set_title('mAP@50', fontsize=12, fontweight='bold')
ax5.set_xlim([0, 1.0])
for i, (bar, val) in enumerate(zip(bars, map50_values)):
    ax5.text(val + 0.02, i, f'{val:.4f}', va='center', fontweight='bold')
ax5.grid(axis='x', alpha=0.3)

# 6. mAP@50-95 comparison
ax6 = fig.add_subplot(gs[2, 1])
map_values = df_results['mAP@50-95'].values
bars = ax6.barh(model_names, map_values, color=colors, alpha=0.8)
ax6.set_xlabel('Score', fontsize=11, fontweight='bold')
ax6.set_title('mAP@50-95', fontsize=12, fontweight='bold')
ax6.set_xlim([0, 1.0])
for i, (bar, val) in enumerate(zip(bars, map_values)):
    ax6.text(val + 0.02, i, f'{val:.4f}', va='center', fontweight='bold')
ax6.grid(axis='x', alpha=0.3)

# 7. Radar chart - Overall performance
ax7 = fig.add_subplot(gs[2, 2], projection='polar')
metrics = df_results.columns.tolist()
angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
angles += angles[:1]

for i, model in enumerate(model_names):
    values = df_results.loc[model].values.tolist()
    values += values[:1]
    ax7.plot(angles, values, 'o-', linewidth=2, label=model, color=colors[i])
    ax7.fill(angles, values, alpha=0.15, color=colors[i])

ax7.set_xticks(angles[:-1])
ax7.set_xticklabels(metrics, size=8)
ax7.set_ylim(0, 1)
ax7.set_title('Overall Performance\n(Radar Chart)', fontsize=12, fontweight='bold', pad=20)
ax7.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
ax7.grid(True)

# Overall title
fig.suptitle('SO SÁNH HIỆU SUẤT CÁC MÔ HÌNH YOLO TRÊN TEST SET', 
             fontsize=16, fontweight='bold', y=0.98)

# Save figure
plot_path = OUTPUT_DIR / "model_comparison.png"
plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='white')
print(f"✅ Đã lưu biểu đồ so sánh vào: {plot_path}")

# Create a simple comparison table visualization
fig2, ax = plt.subplots(figsize=(12, 4))
ax.axis('tight')
ax.axis('off')

# Prepare table data
table_data = []
table_data.append(['Metric'] + model_names)
for metric in df_results.columns:
    row = [metric]
    for model in model_names:
        value = df_results.loc[model, metric]
        row.append(f'{value:.4f}')
    table_data.append(row)

# Highlight best values
cell_colors = [['lightgray'] * (len(model_names) + 1)]
for i, metric in enumerate(df_results.columns):
    row_colors = ['lightgray']
    best_idx = df_results[metric].values.argmax()
    for j in range(len(model_names)):
        if j == best_idx:
            row_colors.append('#90EE90')  # Light green for best
        else:
            row_colors.append('white')
    cell_colors.append(row_colors)

table = ax.table(cellText=table_data, cellColours=cell_colors,
                cellLoc='center', loc='center', 
                colWidths=[0.3] + [0.23] * len(model_names))
table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1, 2.5)

# Style header
for i in range(len(model_names) + 1):
    cell = table[(0, i)]
    cell.set_facecolor('#4472C4')
    cell.set_text_props(weight='bold', color='white')

plt.title('Bảng So Sánh Chi Tiết (Ô màu xanh = giá trị tốt nhất)', 
          fontsize=14, fontweight='bold', pad=20)

table_plot_path = OUTPUT_DIR / "comparison_table.png"
plt.savefig(table_plot_path, dpi=300, bbox_inches='tight', facecolor='white')
print(f"✅ Đã lưu bảng so sánh vào: {table_plot_path}")

print("\n" + "=" * 70)
print("✨ HOÀN THÀNH ĐÁNH GIÁ VÀ SO SÁNH!")
print(f"📁 Tất cả kết quả đã được lưu tại: {OUTPUT_DIR}")
print("=" * 70)

plt.show()
