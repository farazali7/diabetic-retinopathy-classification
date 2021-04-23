import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve, auc, cohen_kappa_score, accuracy_score, confusion_matrix, classification_report
import seaborn as sns
from sklearn.preprocessing import label_binarize
from tensorflow.keras.models import Sequential, load_model, Model
from tqdm import tqdm
import cv2
import pickle

TEST_SIZE = 1928

test_df = pd.read_csv('2015/test_labels.csv')
test_df = test_df.sample(n=TEST_SIZE, random_state=2)


def resize_image(image):
    return cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)


# Normalize to range [0, 1]
def normalize_image(image):
    image = resize_image(image)
    return (image - np.min(image)) / (np.max(image) - np.min(image))


PATH_TO_DATA = '2015/data/test'
x_test = np.empty((test_df.shape[0], 224, 224, 3), dtype=np.float32)
for idx, image_path in enumerate(tqdm(test_df['image'])):
    image = normalize_image(cv2.cvtColor(cv2.imread(f'{PATH_TO_DATA}/{image_path}.jpeg'), cv2.COLOR_BGR2RGB))
    x_test[idx, :, :, :] = image

print(x_test.shape)

model = load_model('models/model-unprocessed-unscaled.h5')
layer_name = 'dense_1'
mobilenetv2_extractor_model = Model(inputs=model.input,
                                    outputs=model.get_layer(layer_name).output)
mobilenetv2_extractor_model.summary()

svm_path = 'models/hybrid-nopca.pkl'
with open(svm_path, 'rb') as fid:
    svm = pickle.load(fid)

y_test_pred = mobilenetv2_extractor_model.predict(x_test)
y_score = np.argmax(svm.decision_function(y_test_pred), axis=1)

y_label = label_binarize(test_df['level'], classes=[0, 1, 2, 3, 4])

kappa_test = cohen_kappa_score(
            np.argmax(y_label, axis=1),
            y_score,
            weights='quadratic'
        )

print('MOBILENETV2-SVM HYBRID NO PCA KAPPA TEST:')
print(kappa_test)
print('--------')

# model = load_model('models/model-unprocessed-unscaled.h5')
# layer_name = 'dense_1'
# mobilenetv2_extractor_model = Model(inputs=model.input,
#                                     outputs=model.get_layer(layer_name).output)
# mobilenetv2_extractor_model.summary()
#
# svm_path = 'models/hybrid-0.9.pkl'
# with open(svm_path, 'rb') as fid:
#     svm = pickle.load(fid)
#
# y_test_pred = mobilenetv2_extractor_model.predict(x_test)
# y_score = np.argmax(svm.decision_function(y_test_pred), axis=1)
#
# y_label = label_binarize(test_df['level'], classes=[0, 1, 2, 3, 4])
#
# kappa_test = cohen_kappa_score(
#             np.argmax(y_label, axis=1),
#             np.argmax(y_score, axis=1),
#             weights='quadratic'
#         )
#
# print('MOBILENETV2-SVM HYBRID NO PCA KAPPA TEST:')
# print(kappa_test)

