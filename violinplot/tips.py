import seaborn as sns

df = sns.load_dataset("tips")
df.violinplot = sns.violinplot(data = df, x="tip", y="sex")




