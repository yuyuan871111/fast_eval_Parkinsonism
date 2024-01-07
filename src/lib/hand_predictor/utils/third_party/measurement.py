# copy-paste from DeepCarc: https://github.com/TingLi2016/DeepCarc
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             matthews_corrcoef, precision_score, recall_score,
                             roc_auc_score)


def measurements(
    y_true,
    y_pred,
    y_pred_prob=None,
    with_auc: bool = True,
    printout: bool = False
):

    acc = accuracy_score(y_true, y_pred)
    sensitivity = recall_score(y_true, y_pred)
    TN, FP, FN, TP = confusion_matrix(y_true, y_pred).ravel()
    specificity = TN / (TN + FP)
    precision = precision_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred)
    npv = TN / (TN + FN)

    if printout:
        print(f"acc: {acc:.2f}")
        print(f"sensitivity: {sensitivity:.2f}")
        print(f"TN: {TN:.0f}; FP: {FP:.0f}; FN: {FN:.0f}; TP: {TP:.0f}")
        print(f"specificity: {specificity:.2f}")
        print(f"precision: {precision:.2f}")
        print(f"f1: {f1:.2f}")
        print(f"mcc: {mcc:.2f}")
        print(f"npv: {npv:.2f}")

    if (with_auc) & (y_pred_prob is not None):
        auc = roc_auc_score(y_true, y_pred_prob)
        if printout:
            print(f"auc: {auc:.2f}")
        return [TN, FP, FN, TP, acc, auc, sensitivity, specificity, precision, npv, f1, mcc]
    else:
        return [TN, FP, FN, TP, acc, sensitivity, specificity, precision, npv, f1, mcc]
