import torch
from sklearn.metrics import precision_recall_fscore_support, average_precision_score, roc_auc_score, precision_recall_curve
import matplotlib.pyplot as plt
import sklearn
import sklearn.metrics
import numpy as np
import pandas as pd
import seaborn as sn


def indices_to_one_hot(data, n_classes=None):
    """ Convert an iterable of indices to one-hot encoded labels.
    

    Parameters
    ----------
    data : list. A list of integers.
    n_classes: int. The number of classes.

    Returns
    -------
    One-hot encoded labels.
    
    """
    targets = np.array(data).reshape(-1)
    targets = targets.astype(np.int8)
    return np.eye(n_classes)[targets]


def calculate_accuracy(outputs, targets):
    with torch.no_grad():
        batch_size = targets.size(0)

        _, pred = outputs.topk(1, 1, largest=True, sorted=True)
        pred = pred.t()
        correct = pred.eq(targets.view(1, -1))
        n_correct_elems = correct.float().sum().item()

        return n_correct_elems / batch_size


def calculate_metrics_for_individual_class(probs, targets, n_classes, label_list=[1]):
    with torch.no_grad():
        _, pred = probs.topk(1, 1, largest=True, sorted=True)
        
        precision, recall, _, _ = precision_recall_fscore_support(
            targets.view(-1, 1).cpu().numpy(),
            pred.cpu().numpy())
        
        target_one_hot = indices_to_one_hot(targets, n_classes=n_classes)
        auc = roc_auc_score(target_one_hot, probs,  average=None)
        aucpr = average_precision_score(target_one_hot, probs, average=None)

        out = {}
        for label in label_list:
            out[label] = [precision[label], recall[label], aucpr[label], auc[label]]
        metrics_names = ['Precision', 'Recall', 'AUCPR', 'AUC']
        return out, metrics_names



def plot_ROC(probs, targets):
    '''
    Generates matplotlib Figure using a trained network, along with images
    and labels from a batch, that shows the network's top prediction along
    with its probability, alongside the actual label, coloring this
    information based on whether the prediction was correct or not.
    Uses the "images_to_probs" function.
    '''
    with torch.no_grad():
        # plot the images in the batch, along with predicted and true labels
        fig = plt.figure(figsize=(6,6))
        fpr, tpr, _ = sklearn.metrics.roc_curve(targets, probs[:,1])

        ax = fig.add_subplot(1, 1, 1)
        ax.plot(fpr, tpr, label='ROC')
        ax.set_ylabel('TPR')
        ax.set_xlabel('FPR')

        return fig



def plot_Recall_Precision_Curve(probs, targets):
    '''
    Generates matplotlib Figure using a trained network, along with images
    and labels from a batch, that shows the network's top prediction along
    with its probability, alongside the actual label, coloring this
    information based on whether the prediction was correct or not.
    Uses the "images_to_probs" function.
    '''
    with torch.no_grad():
        # plot the images in the batch, along with predicted and true labels
        fig = plt.figure(figsize=(6,6))
        precision, recall, _ = precision_recall_curve(targets, probs[:,1])

        ax = fig.add_subplot(1, 1, 1)
        ax.step(recall, precision, where='post')
        ax.set_ylabel('Precision')
        ax.set_xlabel('Recall')
        
        return fig


def plot_confusion_matrix(probs, targets, n_classes):
    with torch.no_grad():
        predictions = np.argmax(probs, axis=-1)
        print(predictions.shape, targets.shape)
        print(predictions[0:10], targets[0:10])
        confusion_mat = sklearn.metrics.confusion_matrix(predictions, targets)
        index_list = [f'Pred_{i}' for i in range(n_classes)]
        column_list = [f'Annot_{i}' for i in range(n_classes)]
        conf_df = pd.DataFrame(confusion_mat, index=index_list, columns=column_list)
        sn.set(font_scale=1.4) # for label size
        plt.figure()
        svm = sn.heatmap(conf_df, annot=True, annot_kws={"size": 16}, fmt="d") # font size
        fig = svm.get_figure()    
        return fig
