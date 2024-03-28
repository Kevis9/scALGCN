import seaborn as sns
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import anndata as ad

# line plot
acc_data = pd.read_csv('../result/stable_acc.csv', index_col=0)

# exp0050
exp0050_projs = [
    'bcp3_6000-bcp2_6000-exp0050_200',
    'bcp3_6000-bcp2_6000-exp0050_400',
    'bcp3_6000-bcp2_6000-exp0050_600',
    'bcp3_6000-bcp2_6000-exp0050_800',
    'bcp3_6000-bcp2_6000-exp0050_1000',    
]

exp0047_projs = [
    'bcp3_6000-bcp2_6000-exp0047_200',
    'bcp3_6000-bcp2_6000-exp0047_400',
    'bcp3_6000-bcp2_6000-exp0047_600',
    'bcp3_6000-bcp2_6000-exp0047_800',
    'bcp3_6000-bcp2_6000-exp0047_1000',        
]

all_y = acc_data.loc[exp0050_projs + exp0047_projs]['GT']   
all_x = [200, 400, 600, 800, 1000] * 2
all_projs = 'exp0050' * len(exp0050_projs) + 'exp0047' * len(exp0047_projs)

res_data = pd.DataFrame({'x': all_x, 'y': all_y, 'proj': all_projs})

sns.lineplot(data=res_data, x='x', y='y', hue='proj')
sns.boxplot(data=res_data,x='x', y='val', hue='x', showfliers=False, width=.3)
plt.xlabel('')
plt.ylabel('')
plt.savefig('lineplot_auxilary_size', dpi=300, transparent=True,bbox_inches="tight")

