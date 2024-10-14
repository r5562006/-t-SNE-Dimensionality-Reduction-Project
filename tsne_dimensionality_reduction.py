# tsne_dimensionality_reduction.py
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# 生成隨機數據
data = np.random.rand(100, 5)

# 應用 t-SNE 降維
tsne = TSNE(n_components=2)
reduced_data = tsne.fit_transform(data)

# 可視化結果
plt.scatter(reduced_data[:, 0], reduced_data[:, 1])
plt.show()