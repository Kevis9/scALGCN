import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

# 所有的EXP0013
exp0013 = [
    "bcp1_6000-bcp2_6000-exp0013",
    "bcp2_6000-bcp1_6000-exp0013",
    "bcp1_6000-bcp3_6000-exp0013",
    "bcp3_6000-bcp1_6000-exp0013",
    "bcp2_6000-bcp3_6000-exp0013",
    "bcp3_6000-bcp2_6000-exp0013",
    "bcp1_6000-pcp1_6000-exp0013",
    "pcp1_6000-bcp1_6000-exp0013",
    "bcp1_6000-mp1_6000-exp0013",
    "mp1_6000-bcp1_6000-exp0013"
]

# 所有的EXP0050
exp0050 = [
    "bcp1_6000-bcp2_6000-exp0050",
    "bcp2_6000-bcp1_6000-exp0050",
    "bcp1_6000-bcp3_6000-exp0050",
    "bcp3_6000-bcp1_6000-exp0050",
    "bcp2_6000-bcp3_6000-exp0050",
    "bcp3_6000-bcp2_6000-exp0050",
    "bcp1_6000-pcp1_6000-exp0050",
    "pcp1_6000-bcp1_6000-exp0050",
    "bcp1_6000-mp1_6000-exp0050",
    "mp1_6000-bcp1_6000-exp0050"
]

# 所有的EXP0047
exp0047 = [
    "bcp1_6000-bcp2_6000-exp0047",
    "bcp2_6000-bcp1_6000-exp0047",
    "bcp1_6000-bcp3_6000-exp0047",
    "bcp3_6000-bcp1_6000-exp0047",
    "bcp2_6000-bcp3_6000-exp0047",
    "bcp3_6000-bcp2_6000-exp0047",
    "bcp1_6000-pcp1_6000-exp0047",
    "pcp1_6000-bcp1_6000-exp0047",
    "bcp1_6000-mp1_6000-exp0047",
]

# 所有的EXP0038
exp0038 = [
    "bcp1_6000-pcp1_6000-exp0038",
    "pcp1_6000-bcp1_6000-exp0038"
]

# 所有的EXP0040
exp0040 = [
    "bcp1_6000-mp1_6000-exp0040",
    "mp1_6000-bcp1_6000-exp0040"
]


acc_data = pd.read_csv('../result/stable_acc.csv', index_col=0)
f1_data = pd.read_csv('../result/stable_macro-f1.csv', index_col=0)

exp0013_acc = acc_data.loc[exp0013]['GT + GL']
exp0050_acc = acc_data.loc[exp0050]['GT + GL']
exp0047_acc = acc_data.loc[exp0047]['GT + GL']
exp0038_acc = acc_data.loc[exp0038]['GT + GL']
exp0040_acc = acc_data.loc[exp0040]['GT + GL']

all_val = exp0013_acc.tolist() + exp0050_acc.tolist() + exp0047_acc.tolist() + exp0038_acc.tolist() + exp0040_acc.tolist()

all_x = ['exp0013'] * len(exp0013_acc) + ['exp0050'] * len(exp0050_acc) + ['exp0047'] * len(exp0047_acc) + ['exp0038'] * len(exp0038_acc) + ['exp0040'] * len(exp0040_acc)

data = pd.DataFrame({'val': all_val, 'x': all_x})

sns.boxplot(data=data,x='x', y='val', hue='x', showfliers=False, width=.3)
plt.xlabel('')
plt.ylabel('')
plt.savefig('exp_detail_boxplot', dpi=300, transparent=True,bbox_inches="tight")
plt.clf()