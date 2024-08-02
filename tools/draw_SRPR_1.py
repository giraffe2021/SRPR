# Separate plots for each dataset including both 5-shot and 1-shot accuracy
import matplotlib.pyplot as plt

# Data for plotting
import numpy as np

meta_inat_mq_5shot = [80.59, 82.04, 82.45, 82.63, 82.69, 82.82, 82.84, 82.83, 82.86, 82.83, 82.81, 82.80, 82.79, 82.78,
                      82.73,
                      82.70, 82.67, 82.61, 82.63, 82.63, 82.64]
meta_inat_mq_1shot = [63.00, 66.22, 67.34, 68.15, 68.59, 68.83, 69.09, 69.35, 69.45, 69.52, 69.58, 69.66, 69.69, 69.76,
                      69.77, 69.80, 69.85, 69.87, 69.89, 69.89, 69.90]
tiere_meta_inat_mq_5shot = [66.91, 67.19, 67.22, 67.29, 67.31, 67.31, 67.31, 67.32, 67.30, 67.24, 67.19, 67.15, 67.10,
                            67.02,
                            67.01, 66.97, 66.95, 66.91, 66.86, 66.84, 66.84]
tiere_meta_inat_mq_1shot = [47.48, 47.64, 47.87, 48.09, 48.19, 48.40, 48.41, 48.48, 48.54, 48.54, 48.56, 48.61, 48.53,
                            48.51, 48.49, 48.45, 48.46, 48.44, 48.37, 48.35, 48.32]
imagenet_mq_5shot = [85.75, 85.87, 85.87, 85.87, 85.87, 85.92, 85.94, 85.99, 86.02, 86.04, 86.07, 86.09, 86.07, 86.06,
                     86.08,
                     86.06, 86.04, 86.02, 86.00, 85.96, 85.92]
imagenet_mq_1shot = [69.25, 69.25, 69.26, 69.31, 69.47, 69.64, 69.77, 69.86, 69.98, 70.09, 70.19, 70.27, 70.32, 70.37,
                     70.38, 70.42, 70.47, 70.44, 70.47, 70.43, 70.42]
# tieredImagenet_mq_5shot = [86.99, 87.32, 87.32, 87.37, 87.44, 87.59, 87.76, 87.90, 88.02, 88.11, 88.21, 88.30, 88.31, 88.34,
#                      88.38, 88.41, 88.38, 88.37, 88.35, 88.33, 88.32]
#
# tieredImagenet_mq_1shot = [68.69, 68.70, 68.80, 69.14, 69.76, 70.56, 71.12, 71.64, 72.10, 72.54, 72.85, 73.11, 73.22,
#                         73.39, 73.50, 73.61, 73.72, 73.79, 73.88, 73.90, 73.91]
tieredImagenet_mq_5shot = list(np.array(
    [0.8699, 0.8737, 0.8737, 0.8739, 0.8743, 0.8751, 0.8763, 0.8773, 0.8785, 0.8800, 0.8811, 0.8822, 0.8829, 0.8833,
     0.8839,
     0.8845, 0.8846, 0.8850, 0.8852, 0.8854, 0.8857]) * 100)

tieredImagenet_mq_1shot = list(np.array(
    [0.6869, 0.6870, 0.6876, 0.6900, 0.6944, 0.7008, 0.7075, 0.7139, 0.7194, 0.7237, 0.7271, 0.7302, 0.7322, 0.7349,
     0.7357, 0.7365, 0.7374, 0.7378, 0.7377, 0.7377, 0.7376]
    ) * 100)

# FC100_mq_5shot = [64.45, 64.65, 64.66, 64.70, 64.73, 64.75, 64.70, 64.71, 64.72, 64.72, 64.70, 64.66, 64.64, 64.62, 64.58,
#             64.56, 64.55, 64.54, 64.53, 64.52, 64.51]
# FC100_mq_1shot = [46.16, 46.18, 46.22, 46.40, 46.54, 46.64, 46.67, 46.71, 46.78, 46.82, 46.88, 46.89, 46.92, 46.94, 46.95,
#                46.94, 46.96, 46.96, 46.98, 47.00, 47.00]

datasets = {
    'meta-iNat': (meta_inat_mq_5shot, meta_inat_mq_1shot),
    'tiered meta-iNat': (tiere_meta_inat_mq_5shot, tiere_meta_inat_mq_1shot),
    'mini-Imagenet': (imagenet_mq_5shot, imagenet_mq_1shot),
    'tiered-Imagenet': (tieredImagenet_mq_5shot, tieredImagenet_mq_1shot),
    # 'FC100': (FC100_mq, FC100_mq_1shot)
}
# Create subplots with the specified modifications
fig, axs = plt.subplots(2, len(datasets), figsize=(len(datasets) * 4, 8))

# Plot each dataset
for index, (label, (mq_data, mq_1s_data)) in enumerate(datasets.items()):
    # 5-shot accuracy
    axs[0, index].plot(mq_data, label=f'{label} 5-shot', color='blue')
    max_idx = mq_data.index(max(mq_data))
    axs[0, index].scatter([max_idx], [max(mq_data)], color='red', marker='x', s=100)  # highlight max value
    axs[0, index].scatter([len(mq_data) - 1], [mq_data[-1]], color='green', marker='o')  # highlight last value
    axs[0, index].text(max_idx, max(mq_data), f'{max(mq_data):.1f}',
                       color='red',
                       fontsize=15)
    axs[0, index].text(len(mq_data) - 1, max(0,mq_data[-1] - 0.07 * (max(mq_data) - min(mq_data))), f'{mq_data[-1]:.1f}',
                       color='green', fontsize=15)
    axs[0, index].set_title(f'{label} 5-shot', fontsize=15)
    if index == 0:
        axs[0, index].set_ylabel('Accuracy (%)', fontsize=15)

    # 1-shot accuracy
    axs[1, index].plot(mq_1s_data, label=f'{label} 1-shot', color='orange')
    max_idx_1s = mq_1s_data.index(max(mq_1s_data))
    axs[1, index].scatter([max_idx_1s], [max(mq_1s_data)], color='red', marker='x', s=100)  # highlight max value
    axs[1, index].scatter([len(mq_1s_data) - 1], [mq_1s_data[-1]], color='green', marker='o')  # highlight last value
    axs[1, index].text(max_idx_1s, max(mq_1s_data),
                       f'{max(mq_1s_data):.1f}', color='red', fontsize=15)
    axs[1, index].text(len(mq_1s_data) - 1, mq_1s_data[-1] - 0.1 * (max(mq_1s_data) - min(mq_1s_data)),
                       f'{mq_1s_data[-1]:.1f}', color='green', fontsize=15)
    axs[1, index].set_title(f'{label} 1-shot', fontsize=15)
    if index == 0:
        axs[1, index].set_ylabel('Accuracy (%)', fontsize=15)

    # Increase x-axis label size
    for ax in axs[:, index]:
        ax.tick_params(axis='x', labelsize=15)
plt.tight_layout(pad=1.1)
plt.show()
fig.savefig("SPRP_in_domain_test.png", dpi=400)
