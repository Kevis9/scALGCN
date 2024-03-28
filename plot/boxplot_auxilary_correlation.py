import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

# 相同类型癌症，aux相关，不相关
same_type_cancer_related = [
    "bcp1_6000-bcp2_6000-exp0013",
    "bcp2_6000-bcp1_6000-exp0013",
    "bcp1_6000-bcp3_6000-exp0013",
    "bcp3_6000-bcp1_6000-exp0013",
    "bcp2_6000-bcp3_6000-exp0013",
    "bcp3_6000-bcp2_6000-exp0013",
]
same_type_cancer_not_related = [
    "bcp1_6000-bcp2_6000-exp0050",
    "bcp2_6000-bcp1_6000-exp0050",
    "bcp1_6000-bcp3_6000-exp0050",
    "bcp3_6000-bcp1_6000-exp0050",
    "bcp2_6000-bcp3_6000-exp0050",
    "bcp3_6000-bcp2_6000-exp0050",
    "bcp1_6000-bcp2_6000-exp0047",
    "bcp2_6000-bcp1_6000-exp0047",
    "bcp1_6000-bcp3_6000-exp0047",
    "bcp3_6000-bcp1_6000-exp0047",
    "bcp2_6000-bcp3_6000-exp0047",
    "bcp3_6000-bcp2_6000-exp0047"
]

diff_type_cancer_ref_related = [
    "bcp1_6000-pcp1_6000-exp0013",
    "pcp1_6000-bcp1_6000-exp0038",
    "bcp1_6000-mp1_6000-exp0013",
    "mp1_6000-bcp1_6000-exp0040"    
]
diff_type_cancer_query_related = [
    "bcp1_6000-pcp1_6000-exp0038",
    "pcp1_6000-bcp1_6000-exp0013",
    "bcp1_6000-mp1_6000-exp0040",
    "mp1_6000-bcp1_6000-exp0013"
]
diff_type_cancer_not_related = [
    "bcp1_6000-pcp1_6000-exp0050",
    "pcp1_6000-bcp1_6000-exp0050",
    "bcp1_6000-mp1_6000-exp0050",
    "mp1_6000-bcp1_6000-exp0050",
    "bcp1_6000-pcp1_6000-exp0047",
    "pcp1_6000-bcp1_6000-exp0047",
    "bcp1_6000-mp1_6000-exp0047",
    "mp1_6000-bcp1_6000-exp0047"
]

acc_data = pd.read_csv('../result/stable_acc.csv', index_col=0)
f1_data = pd.read_csv('../result/stable_macro-f1.csv', index_col=0)

same_type_cancer_related_acc = acc_data.loc[same_type_cancer_related]['GT + GL']
same_type_cancer_not_related_acc = acc_data.loc[same_type_cancer_not_related]['GT + GL']

diff_type_cancer_ref_related_acc = acc_data.loc[diff_type_cancer_ref_related]['GT + GL']
diff_type_cancer_query_related_acc = acc_data.loc[diff_type_cancer_query_related]['GT + GL']
diff_type_cancer_not_related_acc = acc_data.loc[diff_type_cancer_not_related]['GT + GL']

all_val = same_type_cancer_related_acc.tolist() + \
            same_type_cancer_not_related_acc.tolist()

# all_metric = ['acc'] * (len(same_type_cancer_related_acc) + len(same_type_cancer_not_related_acc))

all_x = ['related'] * (len(same_type_cancer_related_acc)) + \
        ['not_related'] * (len(same_type_cancer_not_related_acc))

data = pd.DataFrame({'val': all_val, 'x': all_x})

sns.boxplot(data=data,x='x', y='val', showfliers=False, width=.3)
# plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))    
plt.xlabel('')
plt.ylabel('')
plt.savefig('same_cancer_boxplot_auxilary_correlation', dpi=300, transparent=True,bbox_inches="tight")    
plt.clf()

all_val = diff_type_cancer_ref_related_acc.tolist() + \
            diff_type_cancer_query_related_acc.tolist() + \
            diff_type_cancer_not_related_acc.tolist()

all_x = ['ref_related'] * (len(diff_type_cancer_ref_related_acc)) + \
        ['query_related'] * (len(diff_type_cancer_query_related_acc)) + \
        ['not_related'] * (len(diff_type_cancer_not_related_acc))

data = pd.DataFrame({'val': all_val, 'x': all_x})

sns.boxplot(data=data,x='x', y='val', showfliers=False, width=.3)
# plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.xlabel('')
plt.ylabel('')
plt.savefig('diff_cancer_boxplot_auxilary_correlation', dpi=300, transparent=True,bbox_inches="tight")
plt.clf()