"""
Small helpers: metrics, plotting utilities.
"""
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt

def print_classification(y_true, y_pred, labels=None):
    print(classification_report(y_true, y_pred, target_names=labels))

def plot_confusion_matrix(y_true, y_pred, labels=None):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5,4))
    im = ax.imshow(cm, interpolation='nearest')
    ax.set_xlabel('Predicted'); ax.set_ylabel('True')
    if labels:
        ax.set_xticks(np.arange(len(labels))); ax.set_xticklabels(labels)
        ax.set_yticks(np.arange(len(labels))); ax.set_yticklabels(labels)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i,j], ha="center", va="center", color="w")
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    return fig
