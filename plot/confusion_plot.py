import seaborn as sns
import pandas as pd
import os
import matplotlib.pyplot as plt
def confusion_matrix(true_labels, pred_labels):
    # Create a mapping from class labels to integers
    classes = sorted(set(true_labels + pred_labels))
    label_to_int = {label: i for i, label in enumerate(classes)}

    # Convert labels to integers
    true_labels_int = [label_to_int[label] for label in true_labels]
    pred_labels_int = [label_to_int[label] for label in pred_labels]

    # Initialize the confusion matrix
    num_classes = len(classes)
    matrix = [[0] * num_classes for _ in range(num_classes)]

    # Fill the confusion matrix
    for true, pred in zip(true_labels_int, pred_labels_int):
        matrix[true][pred] += 1

    # Normalize the confusion matrix
    for i in range(num_classes):
        row_sum = sum(matrix[i])
        if row_sum != 0:
            matrix[i] = [count / row_sum for count in matrix[i]]

    # Convert to DataFrame with class labels
    classes = [label for label, _ in sorted(label_to_int.items(), key=lambda x: x[1])]
    conf_matrix_df = pd.DataFrame(matrix, index=classes, columns=classes)
        
    return conf_matrix_df

def read_pred_true_label(res_path):
    preds = pd.read_csv(os.path.join(res_path, 'query_pred.csv')).iloc[:, 0].tolist()
    trues = pd.read_csv(os.path.join(res_path, 'query_true.csv')).iloc[:, 0].tolist()
    return preds, trues

    
result_paths = [
    '/home/Users/kevislin/scALGCN/result/bcp2_6000-bcp3_6000-exp0047_GT + pos + AL + GL',
    '/home/Users/kevislin/scALGCN/other_methods/r_methods/result/bcp2_6000-bcp3_6000/seurat',
    '/home/Users/kevislin/scALGCN/other_methods/r_methods/result/bcp2_6000-bcp3_6000/chetah',
    '/home/Users/kevislin/scALGCN/other_methods/r_methods/result/bcp2_6000-bcp3_6000/scmap'    
]

methods = [
    'scALGT',
    'seurat',
    'chetah',
    'scmap'
]

for i, res_path in enumerate(result_paths):
    preds, trues = read_pred_true_label(res_path=res_path)    
    conf_matrix = confusion_matrix(trues, preds)
    
    sns.heatmap(conf_matrix,linewidths=.5, cbar_kws={"shrink": .5})
    plt.savefig(methods[i]+'_'+'confmatrix', dpi=300, transparent=True)