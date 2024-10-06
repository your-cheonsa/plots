import seaborn as sns
import matplotlib.pyplot as plt #for changing chart size


datasets = sns.get_dataset_names() #sample dataset from seaborn

for dataset_name in datasets: #load the first 5 rows of each dataset
    print("\n Dataset name:" + dataset_name)
    df = sns.load_dataset(dataset_name)
    print(df.head())
    

#set chart size
plt.figure(figsize=(15, 8))

#load dataset
flight_dataset= sns.load_dataset("flights") 

#takes a year value x - (x+9)
decades = lambda x: f"{int(x)}–{int(x + 9)}"

#lambda x : decades(x//10*10) rounds down to the nearest decade
#inner = "box" miniature box-and-whisker plot
passengers_year = sns.violinplot(x = flight_dataset["year"].apply(lambda x : decades(x//10*10)), y = flight_dataset['passengers'], palette = 'pastel', inner = "box", linewidth = 0.8, inner_kws=dict(box_width=7, whis_width=2, color="purple"))


#inner = "quart" quartiles of the data
#passengers_year = sns.violinplot(x = flight_dataset["year"].apply(lambda x : decades(x//10*10)), y = flight_dataset['passengers'], palette = 'pastel', inner = "quart", linewidth = 0.8)



#edit fontsizes of labels and values
plt.xlabel('year', fontsize=8)
plt.ylabel('passengers', fontsize=8)
plt.tick_params(axis='both', labelsize=8)

#show plot
plt.show(passengers_year)

#width corresponds to the approximate frequency of data points in each region
#wider sections = high probability
#narrow sections = low probability

