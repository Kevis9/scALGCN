from learned_graph import get_adjacent_matrix

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

def check_out_similarity_matrix(sm, labels, sm_name):
    types = sorted(list(set(labels)))

    confusion_matrix = []
    for i in range(len(set(types))):
        confusion_matrix.append([0 for j in range(len(types))])

    for i, label in enumerate(types):
        idx = np.where(labels == label)
        label_sm = sm[idx[0], :]
        sm_sum = label_sm.sum(axis=0)
        # print("For {:}({:}), his neighbor situation is".format(label, len(idx[0])))
        for j, type_x in enumerate(types):
            type_x_idx = np.where(labels == type_x)
            # print("{:}: {:} egdes".format(type_x, sum(sm_sum[type_x_idx])))
            confusion_matrix[i][j] = sum(sm_sum[type_x_idx])

    confusion_mat = np.array(confusion_matrix)
    confusion_mat = confusion_mat / np.sum(confusion_mat, axis=1).reshape(-1, 1)
    data_df = pd.DataFrame(
        confusion_mat
    )
    data_df.columns = types
    data_df.index = types

    sns.heatmap(data=data_df, cmap="Blues", cbar=True, xticklabels=True, yticklabels=True)
    plt.savefig(sm_name, dpi=300, bbox_inches="tight")
    plt.clf()


data = pd.read_csv('experiment/cel_seq/cel_seq2_data.csv', index_col=0).to_numpy()
label = pd.read_csv('experiment/cel_seq/cel_seq2_label.csv').to_numpy()

A = get_adjacent_matrix(data)
check_out_similarity_matrix(A, label, 'cel_seq')

