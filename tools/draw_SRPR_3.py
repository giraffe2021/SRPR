import matplotlib.pyplot as plt
import numpy as np

# Data for different shot numbers
shot_1 = [0.6869, 0.6870, 0.6876, 0.6900, 0.6944, 0.7008, 0.7075, 0.7139, 0.7194, 0.7237, 0.7271, 0.7302, 0.7322, 0.7349, 0.7357, 0.7365, 0.7374, 0.7378, 0.7377, 0.7377, 0.7376]
shot_3 = [0.8218, 0.8252, 0.8258, 0.8270, 0.8285, 0.8297, 0.8320, 0.8335, 0.8353, 0.8366, 0.8382, 0.8393, 0.8397, 0.8401, 0.8407, 0.8413, 0.8419, 0.8421, 0.8426, 0.8428, 0.8428]
shot_5 = [0.8699, 0.8737, 0.8737, 0.8739, 0.8743, 0.8751, 0.8763, 0.8773, 0.8785, 0.8800, 0.8811, 0.8822, 0.8829, 0.8833, 0.8839, 0.8845, 0.8846, 0.8850, 0.8852, 0.8854, 0.8857]
shot_10 = [0.8939, 0.8957, 0.8959, 0.8962, 0.8970, 0.8979, 0.8988, 0.8992, 0.9000, 0.9006, 0.9008, 0.9015, 0.9019, 0.9025, 0.9023, 0.9024, 0.9024, 0.9025, 0.9027, 0.9030, 0.9032]
shot_15 = [0.9043, 0.9057, 0.9058, 0.9059, 0.9064, 0.9070, 0.9078, 0.9084, 0.9091, 0.9096, 0.9101, 0.9103, 0.9107, 0.9110, 0.9112, 0.9111, 0.9111, 0.9112, 0.9115, 0.9118, 0.9117]
shot_20 = [0.9087, 0.9103, 0.9103, 0.9106, 0.9109, 0.9114, 0.9115, 0.9116, 0.9124, 0.9129, 0.9130, 0.9136, 0.9140, 0.9143, 0.9145, 0.9149, 0.9149, 0.9150, 0.9153, 0.9153, 0.9154]

iterations = range(21)
final_acc = [shot_1[-1], shot_3[-1], shot_5[-1], shot_10[-1], shot_15[-1], shot_20[-1]]
initial_acc = [shot_1[0], shot_3[0], shot_5[0], shot_10[0], shot_15[0], shot_20[0]]
shot_numbers = ['1', '3', '5', '10', '15', '20']

# Plotting
fig, ax1 = plt.subplots(figsize=(12, 8))


# Bar plots for initial and final accuracy
bar_width = 0.3
index = np.arange(len(shot_numbers))
bar1 = ax1.bar(index - bar_width/2, initial_acc, bar_width, label='Initial Acc', color='gray', alpha=0.7)
bar2 = ax1.bar(index + bar_width/2, final_acc, bar_width, label='Final Acc', color='orange', alpha=0.7)

# Annotations for bars
for rect in bar1 + bar2:
    height = rect.get_height()
    ax1.annotate(f'{height:.4f}',
                 xy=(rect.get_x() + rect.get_width() / 2, height),
                 xytext=(0, 3),  # 3 points vertical offset
                 textcoords="offset points",
                 ha='center', va='bottom')

ax1.tick_params(axis='x', labelsize=15)
ax1.tick_params(axis='y', labelsize=15)
ax1.set_title('Accuracy Over Iterations for Different Shot Numbers', fontsize=18)
ax1.set_xlabel('Iteration', fontsize=15)
ax1.set_ylabel('Accuracy (%)', fontsize=15)
ax1.set_xticks(range(0, 21, 5))
ax1.legend(fontsize=12)

# Adding grid for better readability
ax1.grid(True, linestyle='--', alpha=0.6)

# Secondary x-axis for bar plots
ax2 = ax1.twiny()
ax2.set_xlim(ax1.get_xlim())
ax2.set_xticks(index)
ax2.set_xticklabels(shot_numbers, fontsize=15)
ax2.set_xlabel('Shot Number', fontsize=15)

plt.tight_layout()
plt.show()
