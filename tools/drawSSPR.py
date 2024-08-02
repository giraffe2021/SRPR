import matplotlib.pyplot as plt

# Data structure for plotting
import numpy as np
from matplotlib.ticker import FuncFormatter

datasets = {
    "Omglot": {
        "5-shot": [0.9100, 0.9101, 0.9102, 0.9095, 0.9089, 0.9080, 0.9070, 0.9064, 0.9057, 0.9050, 0.9048, 0.9046,
                   0.9045, 0.9047, 0.9045, 0.9045, 0.9044, 0.9044, 0.9044, 0.9042, 0.9042],
        "1-shot": [0.7538, 0.7539, 0.7543, 0.7538, 0.7537, 0.7527, 0.7516, 0.7508, 0.7501, 0.7499, 0.7497, 0.7496,
                   0.7497, 0.7499, 0.7502, 0.7504, 0.7507, 0.7507, 0.7507, 0.7509, 0.7510]
    },
    "Acraft": {
        "5-shot": [0.4154, 0.4152, 0.4152, 0.4152, 0.4152, 0.4153, 0.4164, 0.4166, 0.4170, 0.4173, 0.4165, 0.4164,
                   0.4164, 0.4163, 0.4161, 0.4160, 0.4159, 0.4158, 0.4156, 0.4157, 0.4158],
        "1-shot": [0.3008, 0.3008, 0.3008, 0.3008, 0.3014, 0.3017, 0.3026, 0.3035, 0.3039, 0.3038, 0.3041, 0.3038,
                   0.3032, 0.3035, 0.3035, 0.3030, 0.3029, 0.3031, 0.3031, 0.3029, 0.3027]
    },
    "CUB": {
        "5-shot": [0.8167, 0.8178, 0.8178, 0.8179, 0.8188, 0.8198, 0.8204, 0.8205, 0.8210, 0.8212, 0.8215, 0.8214,
                   0.8214, 0.8214, 0.8214, 0.8215, 0.8212, 0.8209, 0.8205, 0.8205, 0.8203],
        "1-shot": [0.5825, 0.5825, 0.5826, 0.5835, 0.5864, 0.5890, 0.5916, 0.5937, 0.5950, 0.5969, 0.5988, 0.6001,
                   0.6014, 0.6016, 0.6019, 0.6024, 0.6025, 0.6024, 0.6027, 0.6027, 0.6026]
    },
    "DTD": {
        "5-shot": [0.6758, 0.6819, 0.6844, 0.6868, 0.6880, 0.6892, 0.6898, 0.6905, 0.6908, 0.6906, 0.6907, 0.6905,
                   0.6906, 0.6900, 0.6899, 0.6895, 0.6892, 0.6888, 0.6887, 0.6886, 0.6882],
        "1-shot": [0.4947, 0.4964, 0.4996, 0.5023, 0.5041, 0.5064, 0.5076, 0.5089, 0.5089, 0.5091, 0.5091, 0.5091,
                   0.5087, 0.5084, 0.5087, 0.5084, 0.5079, 0.5079, 0.5077, 0.5078, 0.5078]
    },
    "Fungi": {
        "5-shot": [0.7088, 0.7107, 0.7108, 0.7117, 0.7130, 0.7140, 0.7143, 0.7147, 0.7151, 0.7152, 0.7149, 0.7154,
                   0.7151, 0.7150, 0.7148, 0.7143, 0.7144, 0.7143, 0.7139, 0.7140, 0.7145],
        "1-shot": [0.4895, 0.4895, 0.4901, 0.4921, 0.4945, 0.4972, 0.4999, 0.5012, 0.5030, 0.5037, 0.5046, 0.5052,
                   0.5054, 0.5054, 0.5051, 0.5055, 0.5052, 0.5051, 0.5058, 0.5055, 0.5059]
    },
    "COCO": {
        "5-shot": [0.6218, 0.6271, 0.6285, 0.6320, 0.6350, 0.6373, 0.6406, 0.6426, 0.6446, 0.6465, 0.6482, 0.6490,
                   0.6489, 0.6493, 0.6496, 0.6500, 0.6508, 0.6512, 0.6514, 0.6517, 0.6521],
        "1-shot": [0.4447, 0.4451, 0.4466, 0.4492, 0.4528, 0.4561, 0.4594, 0.4631, 0.4654, 0.4672, 0.4692, 0.4703,
                   0.4712, 0.4720, 0.4729, 0.4731, 0.4736, 0.4739, 0.4747, 0.4752, 0.4759]
    },
    "QDraw": {
        "5-shot": [0.7466, 0.7468, 0.7462, 0.7456, 0.7440, 0.7426, 0.7418, 0.7413, 0.7407, 0.7401, 0.7402, 0.7398,
                   0.7392, 0.7385, 0.7380, 0.7373, 0.7368, 0.7365, 0.7364, 0.7362, 0.7360],
        "1-shot": [0.5584, 0.5591, 0.5601, 0.5596, 0.5601, 0.5603, 0.5598, 0.5595, 0.5595, 0.5594, 0.5597, 0.5598,
                   0.5600, 0.5603, 0.5604, 0.5605, 0.5605, 0.5604, 0.5603, 0.5602, 0.5603]
    },
    "Flower": {
        "5-shot": [0.9360, 0.9366, 0.9368, 0.9371, 0.9376, 0.9378, 0.9382, 0.9385, 0.9385, 0.9382, 0.9380, 0.9380,
                   0.9382, 0.9378, 0.9376, 0.9374, 0.9374, 0.9374, 0.9373, 0.9371, 0.9369],
        "1-shot": [0.7759, 0.7759, 0.7764, 0.7791, 0.7818, 0.7839, 0.7856, 0.7856, 0.7862, 0.7863, 0.7861, 0.7859,
                   0.7854, 0.7848, 0.7845, 0.7838, 0.7835, 0.7828, 0.7826, 0.7822, 0.7821]
    },
    "Sign": {
        "5-shot": [0.7693, 0.7705, 0.7708, 0.7717, 0.7728, 0.7729, 0.7727, 0.7728, 0.7729, 0.7724, 0.7724, 0.7722,
                   0.7723, 0.7720, 0.7719, 0.7717, 0.7717, 0.7714, 0.7710, 0.7709, 0.7707],
        "1-shot": [0.6036, 0.6042, 0.6054, 0.6062, 0.6072, 0.6082, 0.6084, 0.6085, 0.6083, 0.6082, 0.6080, 0.6076,
                   0.6073, 0.6070, 0.6071, 0.6070, 0.6071, 0.6073, 0.6070, 0.6071, 0.6071]
    },
    "_": {
        "5-shot": {},
        "1-shot": {}
    }
    # Additional datasets can be added similarly
}

fig, axs = plt.subplots(4, 5, figsize=(5 * 4, 8))

# Plotting
for index, (label, (mq_data, mq_1s_data)) in enumerate(datasets.items()):
    # Normalize and scale data if necessary
    row = index // 5
    col = index % 5
    if (index+1) % 10 == 0:

        continue
    mq_data = list(np.array(datasets[label][mq_data]) * 100)
    mq_1s_data = list(np.array(datasets[label][mq_1s_data]) * 100)


    axs[row, col].plot(mq_data, label=f'{label} 5-shot', color='blue')
    max_idx = mq_data.index(max(mq_data))
    axs[row, col].scatter([max_idx], [max(mq_data)], color='red', marker='x')  # highlight max value
    axs[row, col].scatter([len(mq_data) - 1], [mq_data[-1]], color='green', marker='o')  # highlight last value
    axs[row, col].text(max_idx, max(mq_data) + 0.01 * (max(mq_data) - min(mq_data)), f'{max(mq_data):.1f}',
                       color='red',
                       fontsize=12)
    if abs(max_idx - len(mq_data)) <= 5:
        axs[row, col].text(len(mq_data) - 1, mq_data[-1] - 0.1 * (max(mq_data) - min(mq_data)), f'{mq_data[-1]:.1f}',
                           color='green', fontsize=12)
    else:
        axs[row, col].text(len(mq_data) - 1, mq_data[-1], f'{mq_data[-1]:.1f}',
                           color='green', fontsize=12)
    axs[row, col].set_title(f'{label} 5-shot', fontsize=15)
    if col == 0:
        axs[row, col].set_ylabel('Accuracy (%)', fontsize=15)

    # 1-shot accuracy
    axs[row + 2, col].plot(mq_1s_data, label=f'{label} 1-shot', color='orange')
    max_idx_1s = mq_1s_data.index(max(mq_1s_data))
    axs[row + 2, col].scatter([max_idx_1s], [max(mq_1s_data)], color='red', marker='x')  # highlight max value
    axs[row + 2, col].scatter([len(mq_1s_data) - 1], [mq_1s_data[-1]], color='green',
                                 marker='o')  # highlight last value
    axs[row + 2, col].text(max_idx_1s, max(mq_1s_data),
                               f'{max(mq_1s_data):.1f}', color='red', fontsize=12)
    if abs(max_idx_1s - len(mq_1s_data)) <=5:
        axs[row + 2, col].text(len(mq_1s_data) - 1, mq_1s_data[-1] - 0.1 * (max(mq_1s_data) - min(mq_1s_data)),
                                   f'{mq_1s_data[-1]:.1f}', color='green', fontsize=12)
    else:
        axs[row + 2, col].text(len(mq_1s_data) - 1, mq_1s_data[-1] ,
                               f'{mq_1s_data[-1]:.1f}', color='green', fontsize=12)
    axs[row + 2, col].set_title(f'{label} 1-shot', fontsize=15)
    if col == 0:
        axs[row + 2, col].set_ylabel('Accuracy (%)', fontsize=15)

    # Increase x-axis label size
    for ax in axs[:, col]:
        ax.tick_params(axis='x', labelsize=15)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y:.1f}'))

# Hide unused subplots
axs[1, 4].axis('off')
axs[3, 4].axis('off')
# Display the plots
plt.tight_layout(pad=1.1, w_pad=0.5, h_pad=1.0)
plt.show()
fig.savefig("SPRP_cross_domain_test.png", dpi=400)
