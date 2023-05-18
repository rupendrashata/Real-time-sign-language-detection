import pandas as pd
import numpy as np
import os
import joblib

from sklearn.preprocessing import LabelBinarizer
from tqdm import tqdm
from imutils import paths

image_paths = list(paths.list_images('../input/preprocessed_image'))

data = pd.DataFrame()

labels = []
for i, image_path in tqdm(enumerate(image_paths), total=len(image_paths)):
    label = image_path.split(os.path.sep)[-2]
    data.loc[i, 'image_path'] = image_path

    labels.append(label)

labels = np.array(labels)
lb = LabelBinarizer()
labels = lb.fit_transform(labels)

print(f"The first one hot encoded labels: {labels[0]}")
print(f"Mapping the first one hot encoded label to its category: {lb.classes_[0]}")
print(f"Total instances: {len(labels)}")

for i in range(len(labels)):
    index = np.argmax(labels[i])
    data.loc[i, 'target'] = int(index)

data = data.sample(frac=1).reset_index(drop=True)

data.to_csv('../input/data.csv', index=False)

print('Saving the binarized labels as pickled file')
joblib.dump(lb, '../outputs/lb.pkl')

print(data.head(10))