import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve, auc, cohen_kappa_score, accuracy_score, confusion_matrix, classification_report
import seaborn as sns
from sklearn.preprocessing import label_binarize
from tensorflow.keras.models import Sequential, load_model
from tqdm import tqdm
import cv2


def normalize_image(image, resize=False):
    if resize:
        image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)
    return (image - np.min(image)) / (np.max(image) - np.min(image))


def show_results(dataframe, model_path, x_val, val_df):
    history = pd.read_csv(dataframe)

    history_acc = history['accuracy']
    history_val_acc = history['val_accuracy']
    history_loss = history['loss']
    history_val_loss = history['val_loss']
    history_val_kappas = history['kappas']

    plt.plot(history_acc)
    plt.plot(history_val_acc)
    plt.legend(['Accuracy', 'Validation Accuracy'])
    plt.show()

    plt.plot(history_loss)
    plt.plot(history_val_loss)
    plt.legend(['Loss', 'Validation Loss'])
    plt.show()

    plt.plot(history_val_kappas)
    plt.legend(['Validation Kappa'], loc='lower right')
    plt.show()

    model = load_model(model_path)
    y_val_pred = model.predict(x_val)
    y_val_pred = np.clip(y_val_pred, 0, 4).astype(int)
    labels = ['0 - No DR', '1 - Mild', '2 - Moderate', '3 - Severe', '4 - Proliferative DR']
    cnf_matrix = confusion_matrix(val_df['diagnosis'].astype('int'), y_val_pred)
    df_cm = pd.DataFrame(cnf_matrix, index=labels, columns=labels)
    plt.figure(figsize=(16, 7))
    sns.heatmap(df_cm, annot=True, cmap="Blues")
    plt.show()

    kappa_val = cohen_kappa_score(
        val_df['diagnosis'].astype('int'),
        y_val_pred,
        weights='quadratic'
    )
    print(kappa_val)

    target_names = ['0 - No DR', '1 - Mild', '2 - Moderate', '3 - Severe', '4 - Proliferative DR']
    print(classification_report(val_df['diagnosis'].astype('int'), y_val_pred, target_names=target_names))


def save_predictions(model_path, output_path, x_val):
    model = load_model(model_path)
    y_val_pred = model.predict(x_val)
    y_val_pred = np.clip(y_val_pred, 0, 4).astype(int)
    df = pd.DataFrame(y_val_pred)
    df.to_csv(output_path, index=False)


def show_history(history_path):
    history = pd.read_csv(history_path)

    history_acc = history['accuracy']
    history_val_acc = history['val_accuracy']
    history_loss = history['loss']
    history_val_loss = history['val_loss']
    history_val_kappas = history['kappas']

    plt.plot(history_acc)
    plt.plot(history_val_acc)
    plt.legend(['Accuracy', 'Validation Accuracy'])
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.show()

    plt.plot(history_loss)
    plt.plot(history_val_loss)
    plt.legend(['Loss', 'Validation Loss'])
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show()

    plt.plot(history_val_kappas)
    plt.legend(['Validation Kappa'], loc='lower right')
    plt.xlabel('Epoch')
    plt.ylabel('Validation Kappa')
    plt.show()


def load_val_data(val_df):
    x_val_raw = np.empty((val_df.shape[0], 224, 224, 3), dtype=np.float32)
    for idx, image_path in enumerate(tqdm(val_df['id_code'])):
        image = normalize_image(cv2.cvtColor(cv2.imread(f'train_images/{image_path}.png'), cv2.COLOR_BGR2RGB),
                                resize=True)
        x_val_raw[idx, :, :, :] = image
    return x_val_raw


def classification_metrics(pred_path, val_df, output_path, isASVM=False):
    y_pred = pd.read_csv(pred_path)
    y_val = val_df['diagnosis'].astype('int')

    if isASVM:
        y_pred = np.array(y_pred.idxmax(axis=1)).astype(int)
        y_val = np.argmax(label_binarize(val_df['diagnosis'], classes=[0,1,2,3,4]), axis=1)

    labels = ['0 - No DR', '1 - Mild', '2 - Moderate', '3 - Severe', '4 - Proliferative DR']
    cnf_matrix = confusion_matrix(y_val, y_pred)
    df_cm = pd.DataFrame(cnf_matrix, index=labels, columns=labels)
    plt.figure(figsize=(70, 30))
    sns.set(font_scale=6.0)  # Adjust to fit
    sns.heatmap(df_cm, annot=True, cmap="Blues", fmt='g')
    plt.xlabel('Predicted', fontsize=80)
    plt.ylabel('Actual', fontsize=80)
    plt.show()

    kappa_val = cohen_kappa_score(
        y_val,
        y_pred,
        weights='quadratic'
    )
    print(kappa_val)

    target_names = ['0 - No DR', '1 - Mild', '2 - Moderate', '3 - Severe', '4 - Proliferative DR']
    print(classification_report(y_val, y_pred, target_names=target_names))
    report = classification_report(y_val, y_pred, target_names=target_names, output_dict=True)
    df = pd.DataFrame(report).transpose()
    df.to_csv(output_path, index=True)


score_mnetsvm = 'history-preds/hybrid-scores-nopca.csv'
score_mnetsvmpca = 'history-preds/hybrid-scores-0.9.csv'
score_svm = 'history-preds/svm-alone-scores-0.9NEW.csv'
score_mnet = 'history-preds/mnetonly-scores.csv'
score_mnetnoaug = 'history-preds/mnetonly-noaugment-scores.csv'

val_df = pd.read_csv('labels/val_data.csv')

classification_metrics(score_mnetsvm, val_df, 'mnetsvm-clfreport.csv', True)
# classification_metrics(score_mnetsvmpca, val_df, 'mnetsvmpca-clfreport.csv', True)
# classification_metrics(score_svm, val_df,'svm-clfreport.csv', True)
# classification_metrics(score_mnet, val_df, 'mnet-clfreport.csv')
# classification_metrics(score_mnetnoaug, val_df, 'mnetnoaug-clfreport.csv')


