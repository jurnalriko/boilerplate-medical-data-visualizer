import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# 1
df = pd.read_csv('medical_examination.csv')

# 2
#df['overweight'] = df['weight'] / (df['height']/100).apply(lambda x: x**2)
df['bmi'] = df['weight'] / (df['height']/100)**2

def kategori_bmi(bmi):
    return 1 if bmi > 25 else 0

df['overweight'] = df['bmi'].map(kategori_bmi)

df = df.drop('bmi', axis=1)

# 3
def normalizing_df(value):
    if value == 1:
        return 0
    elif value > 1:
        return 1
    else:
        return value

df['cholesterol'] = df['cholesterol'].map(normalizing_df)
df['gluc'] = df['gluc'].map(normalizing_df)



# 4
def draw_cat_plot():
    # 5
    df_cat = pd.melt(df, id_vars=['cardio'], value_vars=['active', 'alco', 'cholesterol', 'gluc', 'overweight', 'smoke'])
    
    # 6
    df_cat = df_cat.groupby(['cardio', 'variable', 'value']).size().reset_index(name='total')
    
    # 7


    # 8
    fig = sns.catplot(data = df_cat, x='variable', y='total', hue='value', col='cardio', kind='bar').fig


    # 9
    fig.savefig('catplot.png')
    return fig


# 10
def draw_heat_map():
    # 11
    df_heat = df[
        (df['ap_lo'] <= df['ap_hi']) &
        (df['height'] >= df['height'].quantile(0.025)) &
        (df['height'] <= df['height'].quantile(0.975)) &
        (df['weight'] >= df['weight'].quantile(0.025)) &
        (df['weight'] <= df['weight'].quantile(0.975))
    ]

    # 12
    corr = df_heat.corr(method='pearson')

    # 13
    mask = np.triu(corr)


    # 14
    fig, ax = plt.subplots(figsize=(12,12))

    # 15
    ax = sns.heatmap(corr, linewidths=1, annot=True, square=True, mask=mask, fmt='.1f', center=0.08, cbar_kws={"shrink":0.5})

    # 16
    fig.savefig('heatmap.png')
    return fig
