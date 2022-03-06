from time import process_time_ns
from sign_dataset import SignDataset
from matplotlib import pyplot as plt
import numpy as np

dataset = SignDataset(root_dir="C:\\Users\\Yannick Wattenberg\\Documents\\repos\\TrafficSignClassification\\data\\Images", csv_file="labels.csv")
image = (dataset[0])[0]
print(image)
image = np.transpose(image, (2,0,1))
print(image)
print((dataset[1000])[1])
plt.show()
