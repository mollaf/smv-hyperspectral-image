


#%%


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


#%%

plt.rcParams['figure.figsize'] = (20.0, 10.0)
plt.rcParams['font.family'] = "serif"

df = pd.pivot_table(data=sns.load_dataset("flights"),
                    index='month',
                    values='passengers',
                    columns='year')
df.head()

plt.figure()
sns.heatmap(df)

plt.figure()
sns.heatmap(df, cmap='coolwarm')

plt.figure()
midpoint = (df.values.max() - df.values.min()) / 2
sns.heatmap(df, cmap='coolwarm', center=midpoint)

plt.figure()
midpoint = (df.values.max() - df.values.min()) / 2
sns.heatmap(df, cmap='coolwarm', center=midpoint, vmin=150, vmax=400)

plt.figure()
p = sns.heatmap(df, cmap='coolwarm', annot=True, fmt=".1f")

plt.figure()
p = sns.heatmap(df,
                cmap='coolwarm',
                annot=True,
                fmt=".1f",
                annot_kws={'size':10},
                cbar=False,
                square=True)













# %%


plt.rcParams['font.size'] = 20
bg_color = (0.88,0.85,0.95)
plt.rcParams['figure.facecolor'] = bg_color
plt.rcParams['axes.facecolor'] = bg_color
fig, ax = plt.subplots(1)
p = sns.heatmap(df,
                cmap='coolwarm',
                annot=True,
                fmt=".1f",
                annot_kws={'size':16},
                ax=ax)
plt.xlabel('Month')
plt.ylabel('Year')
ax.set_ylim((0,15))
plt.text(5,12.3, "Heat Map", fontsize = 95, color='Black', fontstyle='italic')











# %%

uniform_data = np.random.rand(10, 12)
plt.figure()
ax = sns.heatmap(uniform_data)

plt.figure()
ax = sns.heatmap(uniform_data, vmin=0, vmax=1)











# %%




















