import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

colors = [
    '8ECFC9',
    'FFBE7A',
    'FA7F6F',
    '82B0D2',
    'BEB8DC',
    'E7DAD2'    
]


def draw_barplot(data, save_path):        
    ax = sns.barplot(x="exp", y="val", hue="method", data=data, palette=sns.color_palette(colors))        
    plt.xticks(rotation=90)
    ax.set_ylabel('')                    
    plt.savefig(save_path, dpi=300, transparent=True, bbox_inches="tight")
    plt.clf()

def to_long_format(data):    
    # data的exp是bcp1_6000-bcp2_6000的格式, 变成BCP1-BCP2_6000，并且转大写
    data['exp'] = data['exp'].apply(lambda x: x.upper())
    data['exp'] = data['exp'].apply(lambda x: x.replace('_6000', ''))    
    data = data.melt(id_vars=['exp'], var_name='method', value_name='val')    
    return data        


acc_data = to_long_format(pd.read_csv('cancer_res_acc_data.csv'))
f1_data = to_long_format(pd.read_csv('cancer_res_f1_data.csv'))

draw_barplot(acc_data, 'cancer_res_acc_bar.png')
draw_barplot(f1_data, 'cancer_res_f1_bar.png')


