import numpy as np

# Load the .npz file
data = np.load('datasets\58758_train.npz')

# See the keys inside (like a dictionary)
print(data.files)

# Access specific array
print(data['test_predictions'])
