import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

# 相同类型癌症，有 auxilary data
same_type_with_aux = [
    "bcp1_6000-bcp2_6000-exp0013",
    "bcp1_6000-bcp2_6000-exp0050",
    "bcp1_6000-bcp2_6000-exp0047",

    "bcp2_6000-bcp1_6000-exp0013",
    "bcp2_6000-bcp1_6000-exp0050",
    "bcp2_6000-bcp1_6000-exp0047",

    "bcp1_6000-bcp3_6000-exp0013",
    "bcp1_6000-bcp3_6000-exp0050",
    "bcp1_6000-bcp3_6000-exp0047",

    "bcp3_6000-bcp1_6000-exp0013",
    "bcp3_6000-bcp1_6000-exp0050",
    "bcp3_6000-bcp1_6000-exp0047",

    "bcp2_6000-bcp3_6000-exp0013",
    "bcp2_6000-bcp3_6000-exp0050",
    "bcp2_6000-bcp3_6000-exp0047",
    
    "bcp3_6000-bcp2_6000-exp0013",
    "bcp3_6000-bcp2_6000-exp0050",
    "bcp3_6000-bcp2_6000-exp0047"
]

# 相同类型癌症，没有 auxilary data
same_type_no_aux = [
    "bcp1_6000-bcp2_6000",
    "bcp2_6000-bcp1_6000",

    "bcp1_6000-bcp3_6000",
    "bcp3_6000-bcp1_6000",

    "bcp2_6000-bcp3_6000",        
    "bcp3_6000-bcp2_6000"
]

# 不同类型癌症，有 auxilary data
diff_type_with_aux = [
    "bcp1_6000-pcp1_6000-exp0013",
    "bcp1_6000-pcp1_6000-exp0038",
    "bcp1_6000-pcp1_6000-exp0050",    
    "bcp1_6000-pcp1_6000-exp0047",

    "pcp1_6000-bcp1_6000-exp0038",
    "pcp1_6000-bcp1_6000-exp0013",
    "pcp1_6000-bcp1_6000-exp0050",
    "pcp1_6000-bcp1_6000-exp0047",

    "bcp1_6000-mp1_6000-exp0013",
    "bcp1_6000-mp1_6000-exp0040",
    "bcp1_6000-mp1_6000-exp0050",
    "bcp1_6000-mp1_6000-exp0047",

    'mp1_6000-bcp1_6000-exp0040',
    'mp1_6000-bcp1_6000-exp0013',
    'mp1_6000-bcp1_6000-exp0050'    
]

# 不同类型癌症，没有 auxilary data
diff_type_no_aux = [
    "bcp1_6000-mp1_6000",
    "mp1_6000-bcp1_6000",
    "bcp1_6000-pcp1_6000",
    "pcp1_6000-bcp1_6000"
]

acc_data = pd.read_csv('../result/stable_acc.csv', index_col=0)
f1_data = pd.read_csv('../result/stable_macro-f1.csv', index_col=0)

same_type_with_aux_acc = acc_data.loc[same_type_with_aux]['GT + GL']
same_type_no_aux_acc = acc_data.loc[same_type_no_aux]['GT + GL']
# same_type_with_aux_f1 = f1_data.loc[same_type_with_aux]['GT + GL']
# same_type_no_aux_f1 = f1_data.loc[same_type_no_aux]['GT + GL']
print("相同类型with aux平均acc:{:.3f}".format(same_type_with_aux_acc.mean()))
print("相同类型no aux平均acc:{:.3f}".format(same_type_no_aux_acc.mean()))


diff_type_with_aux_acc = acc_data.loc[diff_type_with_aux]['GT + GL']
diff_type_no_aux_acc = acc_data.loc[diff_type_no_aux]['GT + GL']

print("不同类型with aux平均acc:{:.3f}".format(diff_type_with_aux_acc.mean()))
print("不同类型no aux平均acc:{:.3f}".format(diff_type_no_aux_acc.mean()))

# diff_type_with_aux_f1 = f1_data.loc[diff_type_with_aux]['GT + GL']
# diff_type_no_aux_f1 = f1_data.loc[diff_type_no_aux]['GT + GL']

# 进行相同癌症的数据整合
all_val = same_type_no_aux_acc.tolist() + \
          same_type_with_aux_acc.tolist() 
        #   same_type_no_aux_f1.tolist() + \
        #     same_type_with_aux_f1.tolist()            
all_metric = ['acc'] * (len(same_type_no_aux_acc) + len(same_type_with_aux_acc))
            #  ['f1'] * (len(same_type_no_aux_f1) + len(same_type_with_aux_f1))

all_x = ['no_aux'] * (len(same_type_no_aux_acc)) + \
        ['with_aux'] * (len(same_type_with_aux_acc)) 
        # ['no_aux'] * (len(same_type_no_aux_f1)) + \
        # ['with_aux'] * (len(same_type_with_aux_f1))

data = pd.DataFrame({'val': all_val, 'x': all_x})
sns.boxplot(data=data,x='x', y='val', hue='x', showfliers=False, width=.3)
# plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))    
plt.xlabel('')
plt.ylabel('')
plt.savefig('same_cancer_boxplot_auxilary', dpi=300, transparent=True,bbox_inches="tight")    
plt.clf()

# 进行不同癌症的数据整合
all_val = diff_type_no_aux_acc.tolist() + \
          diff_type_with_aux_acc.tolist()
        #   diff_type_no_aux_f1.tolist() + \
        #     diff_type_with_aux_f1.tolist()
all_metric = ['acc'] * (len(diff_type_no_aux_acc) + len(diff_type_with_aux_acc))
            #  ['f1'] * (len(diff_type_no_aux_f1) + len(diff_type_with_aux_f1))

all_x = ['no_aux'] * (len(diff_type_no_aux_acc)) + \
        ['with_aux'] * (len(diff_type_with_aux_acc))
        # ['no_aux'] * (len(diff_type_no_aux_f1)) + \
        # ['with_aux'] * (len(diff_type_with_aux_f1))

data = pd.DataFrame({'val': all_val, 'x': all_x})

sns.boxplot(data=data,x='x', y='val', hue='x', showfliers=False, width=.3)

# plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.xlabel('')
plt.ylabel('')
plt.savefig('diff_cancer_boxplot_auxilary', dpi=300, transparent=True,bbox_inches="tight")
plt.clf()