
import os
import random
import numpy as np
from PIL import Image

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
import seaborn as sns
import warnings
import collections
import json
from sklearn.model_selection import StratifiedKFold, train_test_split, ShuffleSplit

from sklearn.metrics import roc_curve, auc, roc_auc_score, precision_recall_curve, average_precision_score
from itertools import cycle


try:
    from joblib import dump, load
except ImportError:
    from sklearn.externals.joblib import dump, load

    
def get_test_scores(model, test_img_list, test_img_path, num_classes):
    num_test = len(test_img_list)
    y_score = np.zeros((num_test, num_classes))
    y_pred = np.zeros(num_test, dtype='int')
    
    for idx, fname in enumerate(test_img_list):
        img_filepath = os.path.join(test_img_path, fname)
        pred_cls, pred_cls_str, pred_score, pred_scores = model.predict(img_filepath, show_image=False)
        y_score[idx, :] = pred_scores 
        y_pred[idx] = pred_cls
        
    return y_score, y_pred


def plot_roc_curve(y_true, y_score, is_single_fig=False):
    """
    Plot ROC Curve and show AUROC score
    """    
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    plt.title('AUROC = {:.4f}'.format(roc_auc))
    plt.plot(fpr, tpr, 'b')
    plt.plot([0,1], [0,1], 'r--')
    plt.xlim([-0.05,1.05])
    plt.ylim([-0.05,1.05])
    plt.ylabel('TPR(True Positive Rate)')
    plt.xlabel('FPR(False Positive Rate)')
    if is_single_fig:
        plt.show()
    
def plot_pr_curve(y_true, y_score, is_single_fig=False):
    """
    Plot Precision Recall Curve and show AUPRC score
    """
    prec, rec, thresh = precision_recall_curve(y_true, y_score)
    avg_prec = average_precision_score(y_true, y_score)
    plt.title('AUPRC = {:.4f}'.format(avg_prec))
    plt.step(rec, prec, color='b', alpha=0.2, where='post')
    plt.fill_between(rec, prec, step='post', alpha=0.2, color='b')
    plt.plot(rec, prec, 'b')
    plt.xlim([-0.05,1.05])
    plt.ylim([-0.05,1.05])
    plt.ylabel('Precision')
    plt.xlabel('Recall')
    if is_single_fig:
        plt.show()

def plot_conf_mtx(y_true, y_score, thresh=0.5, class_labels=['0','1'], is_single_fig=False):
    """
    Plot Confusion matrix
    """    
    y_pred = np.where(y_score >= thresh, 1, 0)
    print("confusion matrix (cutoff={})".format(thresh))
    print(classification_report(y_true, y_pred, target_names=class_labels))
    conf_mtx = confusion_matrix(y_true, y_pred)
    sns.heatmap(conf_mtx, xticklabels=class_labels, yticklabels=class_labels, annot=True, fmt='d', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.ylabel('True Class')
    plt.xlabel('Predicted Class')
    if is_single_fig:
        plt.show()

def plot_roc_curve_multiclass(y_true_ohe, y_score, num_classes, color_table, skip_legend=5, is_single_fig=False):
    """
    Plot ROC curve to multi-class
    """     
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(num_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_ohe[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_true_ohe.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    
    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(num_classes)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(num_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= num_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])  
    
    colors = cycle(color_table)
    
    fig, ax = plt.subplots(figsize=(8,8))
    ax.plot(fpr["micro"], tpr["micro"],
            label='micro-average ROC curve (area = {0:0.5f})'.format(roc_auc["micro"]),
            color='deeppink', linewidth=3)

    ax.plot(fpr["macro"], tpr["macro"],
            label='macro-average ROC curve (area = {0:0.5f})'.format(roc_auc["macro"]),
            color='navy', linewidth=3)

    for i, color in zip(range(num_classes), colors):
        if i % skip_legend == 0:
            label='ROC curve of class {0} (area = {1:0.4f})'.format(i, roc_auc[i])
        else:
            label=None
        ax.plot(fpr[i], tpr[i], color=color, label=label, lw=2, alpha=0.3, linestyle=':')
        ax.grid(alpha=.4)

    ax.plot([0, 1], [0, 1], 'k--', lw=2)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Receiver operating characteristic to multi-class')
    ax.legend(loc="lower right", prop={'size':10})
    if is_single_fig:
        plt.show()    
        
        
def plot_pr_curve_multiclass(y_true_ohe, y_score, num_classes, color_table, skip_legend=5, is_single_fig=False):
    """
    Plot precision-recall curve to multi-class
    """      
    precision = dict()
    recall = dict()
    average_precision = dict()
    for i in range(num_classes):
        precision[i], recall[i], _ = precision_recall_curve(y_true_ohe[:, i], y_score[:, i])
        average_precision[i] = average_precision_score(y_true_ohe[:, i], y_score[:, i])

    # A "micro-average": quantifying score on all classes jointly
    precision["micro"], recall["micro"], _ = precision_recall_curve(y_true_ohe.ravel(), y_score.ravel())
    average_precision["micro"] = average_precision_score(y_true_ohe, y_score, average="micro")
    average_precision["macro"] = average_precision_score(y_true_ohe, y_score, average="macro")

    all_precision = np.unique(np.concatenate([precision[i] for i in range(num_classes)]))

    # Then interpolate all ROC curves at this points
    mean_recall = np.zeros_like(all_precision)
    for i in range(num_classes):
        mean_recall += np.interp(all_precision, precision[i], recall[i])

    # Finally average it and compute AUC
    mean_recall /= num_classes

    precision["macro"] = all_precision
    recall["macro"] = mean_recall

    colors = cycle(color_table)
    fig, ax = plt.subplots(figsize=(8,8))    
    label = 'micro-average Precision-recall (area = {0:0.4f})'.format(average_precision["micro"])
    ax.plot(recall["micro"], precision["micro"], label=label, color='deeppink', lw=3)

    label = 'macro-average Precision-recall (area = {0:0.4f})'.format(average_precision["macro"])
    ax.plot(recall["macro"], precision["macro"], label=label, color='navy', lw=3)

    for i, color in zip(range(num_classes), colors):
        if i % skip_legend == 0:
            label = 'PR for class {0} (area = {1:0.4f})'.format(i, average_precision[i])
        else:
            label = None
        ax.plot(recall[i], precision[i], color=color, label=label, lw=2, alpha=0.5, linestyle=':')  

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Precision-Recall curve to multi-class')
    ax.legend(loc="lower left", prop={'size':10})
   
    if is_single_fig:
        plt.show()  
        
        
def plot_conf_mtx_multiclass(y_true, y_pred, labels, is_single_fig=False):
    """
    Plot confusion matrix to multi-class
    """         
    from sklearn.metrics import confusion_matrix, classification_report
    import seaborn as sns
    
    cm = confusion_matrix(y_true, y_pred) 
    print(classification_report(y_true, y_pred, target_names=labels))
    plt.figure(figsize=(14,12))
    plt.title('Confusion Matrix')    
    sns.heatmap(cm, xticklabels=labels, yticklabels=labels, annot=True, fmt='d', cmap=plt.cm.Blues)
    
    if is_single_fig:
        plt.show()                     