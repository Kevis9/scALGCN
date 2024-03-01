import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('al_exp.csv', sep='\t', index_col=0)
sns.boxplot(x='method', y='val', hue='metric', howfliers=False)
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))    
plt.savefig('active_learning_box_plot', dpi=300, transparent=True,bbox_inches="tight")    
