import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('diff_cancer_exp.csv', sep='\t')
sns.boxplot(data=data,x='method', y='val', hue='metric', showfliers=False, width=.5)
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))    
plt.xlabel('')
plt.ylabel('')
plt.savefig('diff_cancer_box_plot', dpi=300, transparent=True,bbox_inches="tight")    
