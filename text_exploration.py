import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from wordcloud import WordCloud
from utils.text_processes import preprocess_text, tokenize_text

# Part1: Load dataset
path = "https://drive.google.com/uc?id=1-pp62M_iZB-3ZZTzMI_TWosJOV-a_6OA"
df = pd.read_csv(path)
null = df.isnull().sum()

print(f'example data:\n{df.head()}')
print(f'shape of data: {df.shape}')
print(f'columns name: {df.columns.values}')
print(f'null data:\n{null}')
print("\n")

# --------------------------------------------------------------------------------------
# Part2: Data Exploration
count = Counter(df['labels'])
total_count = df.shape[0]
print(f'Distribution each categories: {count}')

# Calculate percentages
percentages = {label: (count_value / total_count) * 100 for label, count_value in count.items()}

# Sort by most common
percentages = dict(sorted(percentages.items(), key=lambda item: item[1], reverse=True))
print(percentages)


# Extract labels and values
labels = list(percentages.keys())
values = list(percentages.values())

# Create a colormap - optional
cmap = plt.get_cmap('viridis')

# Generate colors from the colormap - optional
colors = cmap(np.linspace(0.4, 0.9, len(percentages)))

# Create a pie chart
plt.figure(figsize=(10, 6))
plt.subplot(1,2,1)

plt.pie(values, labels=labels, autopct='%1.2f%%', startangle=90, colors=colors)
plt.title('BBC News Categories Distribution (Percentages)')
plt.axis('equal')

# Create a df for bar chart
df_percentages = pd.DataFrame(list(percentages.items()), columns=['labels', 'Percentage'])

# Plot the bar chart
plt.subplot(1,2,2)
ax = sns.barplot(x='labels', y='Percentage', hue=np.linspace(0.4, 0.9, len(percentages)),
            data=df_percentages, palette='viridis', legend=False)

# Add data labels (annotations) on top of the bars
for x,y in np.stack((df_percentages['labels'],df_percentages['Percentage']), axis=1):
    ax.annotate(f'{y:.2f}%', (x,y), ha= 'center',color="w",xytext=(x,y-(y/2)))

# Set labels and title
plt.title('BBC News Categories Distribution (Percentages)')
plt.ylabel('Percentage')
plt.xlabel('Categories')

# Show plot
plt.tight_layout()
plt.show()

# --------------------------------------------------------------------------------------
# Part3: text processing
# Apply the preprocessing function to the 'text' column : tokenization, lower cases, stopword and stemming
df['processed_words'] = df['data'].apply(preprocess_text)
# print(df[['data', 'processed_words']][0:5])

# Apply the preprocessing function to the 'text' column : tokenization and lower cases - optional
df['tokenize_words'] = df['data'].apply(tokenize_text)
# print(df[['data', 'tokenize_words']][0:5])

# --------------------------------------------------------------------------------------
# Part4: Generate Wordcloud
categories = ['entertainment' ,'business' ,'sport' ,'politics', 'tech']
words_dict = {}

# Step4-1: Explode word each categories :
for category in categories:
    # Filter rows based on categories
    filtered_rows = df.loc[df['labels'] == category]

    # Explode the 'processed_words' and 'tokenize_words' columns
    words_dict[f'{category}_words_pro'] = filtered_rows.explode('processed_words').reset_index(drop=True)
    words_dict[f'{category}_words_tok'] = filtered_rows.explode('tokenize_words').reset_index(drop=True)


# Step4-2: access the exploded words with:
words_ent = words_dict['entertainment_words_pro']['processed_words']
words_bus = words_dict['business_words_pro']['processed_words']
words_spt = words_dict['sport_words_pro']['processed_words']
words_pol = words_dict['politics_words_pro']['processed_words']
words_tch = words_dict['tech_words_pro']['processed_words']

# Step4-3: Generate word cloud
# Convert the 'processed_words' column to a single string separate with ' '
words_ent = ' '.join(words_ent.dropna().astype(str))
words_bus = ' '.join(words_bus.dropna().astype(str))
words_spt = ' '.join(words_spt.dropna().astype(str))
words_pol = ' '.join(words_pol.dropna().astype(str))
words_tch = ' '.join(words_tch.dropna().astype(str))

# Generate word cloud
wordcloud_ent = WordCloud(width=800, height=400, background_color='white').generate(words_ent)
wordcloud_bus = WordCloud(width=800, height=400, background_color='white').generate(words_bus)
wordcloud_spt = WordCloud(width=800, height=400, background_color='white').generate(words_spt)
wordcloud_pol = WordCloud(width=800, height=400, background_color='white').generate(words_pol)
wordcloud_tch = WordCloud(width=800, height=400, background_color='white').generate(words_tch)

# Step4-4: Plotting
wordclouds = [wordcloud_ent, wordcloud_bus, wordcloud_spt, wordcloud_pol, wordcloud_tch]

# Create the figure
fig = plt.figure(figsize=(10, 10))

# Create the grid layout using subplot2grid
axs = [plt.subplot2grid((3, 2), (i//2, i%2), colspan=2 if i == 4 else 1) for i in range(len(categories))]

# Loop over the word clouds and their corresponding axes
for i, ax in enumerate(axs):
    ax.imshow(wordclouds[i], interpolation='bicubic')
    ax.axis('off')
    ax.set_title(f'Word Cloud {categories[i]}')

# Adjust the spacing - optional
plt.subplots_adjust(top=0.9, bottom=0.1, right=0.95, left=0.05)

# Display the plots
plt.show()

# --------------------------------------------------------------------------------------
# Part5: Generate Frequency word each category
# Calculate word frequencies
# Split text into individual words
words_ent_split_freq = Counter(words_ent.split())
words_bus_split_freq = Counter(words_bus.split())
words_spt_split_freq = Counter(words_spt.split())
words_pol_split_freq = Counter(words_pol.split())
words_tch_split_freq = Counter(words_tch.split())

print(f'example number unique words in tech: {len(words_tch_split_freq.most_common())}')
print(f'total words in tech: {sum(words_tch_split_freq.values())}')

# Convert the word frequencies to DataFrames
df_word_freq_ent = pd.DataFrame(words_ent_split_freq.items(), columns=['Word', 'Frequency']).sort_values(by='Frequency', ascending=False)
df_word_freq_bus = pd.DataFrame(words_bus_split_freq.items(), columns=['Word', 'Frequency']).sort_values(by='Frequency', ascending=False)
df_word_freq_spt = pd.DataFrame(words_spt_split_freq.items(), columns=['Word', 'Frequency']).sort_values(by='Frequency', ascending=False)
df_word_freq_pol = pd.DataFrame(words_pol_split_freq.items(), columns=['Word', 'Frequency']).sort_values(by='Frequency', ascending=False)
df_word_freq_tch = pd.DataFrame(words_tch_split_freq.items(), columns=['Word', 'Frequency']).sort_values(by='Frequency', ascending=False)

# List of dataframes and categories
dataframes = [df_word_freq_ent, df_word_freq_bus, df_word_freq_spt, df_word_freq_pol, df_word_freq_tch]
colors = ['skyblue', 'lightgreen', 'salmon', 'paleturquoise', 'pink']

# Create subplots with 1 row and 5 columns
fig, axs = plt.subplots(1, 5, figsize=(18, 6))  # 1 row, 5 columns

# Loop over the dataframes and corresponding categories
for i, (df_i, color, category) in enumerate(zip(dataframes, colors, categories)):
    axs[i].barh(df_i.head(20)['Word'], df_i.head(20)['Frequency'], color=color)
    axs[i].set_xlabel('Frequency')
    axs[i].set_title(f'Top 20 Words - {category}')
    axs[i].invert_yaxis()  # Invert y-axis for the horizontal bar plot

# Adjust layout and spacing
plt.tight_layout()

# Show the plot
plt.show()
# --------------------------------------------------------------------------------------
# Part6: Create Text length Chart
df['data'] = df['data'].astype(str)

# Calculate sentence lengths (in terms of word count)
df_text_length = df
for category in categories:
    # Filter rows based on categories and split
    df_text_length[f'text_length_{category}'] = df.loc[df['labels'] == f'{category}', 'data'].apply(lambda x: len(x.split()))

max_ent = int(df_text_length['text_length_entertainment'].max())
max_bus = int(df_text_length['text_length_business'].max())
max_spt = int(df_text_length['text_length_sport'].max())
max_pol = int(df_text_length['text_length_politics'].max())
max_tch = int(df_text_length['text_length_tech'].max())

# Plotting
max_lengths_list = [max_ent, max_bus, max_spt, max_pol, max_tch]
text_length_list = ["text_length_entertainment",
                    "text_length_business",
                    "text_length_sport",
                    "text_length_politics",
                    "text_length_tech"]

fig, axs = plt.subplots(1, 5, figsize=(18, 6), sharey=True)
x = 100

# Loop over the dataframes and corresponding categories
for i, (max_tl, tl, category) in enumerate(zip(max_lengths_list, text_length_list, categories)):
    axs[i].hist(df_text_length[f'{tl}']/x, bins=range(1, int(max_tl/100) + 2), edgecolor='black')
    axs[i].set_xlabel('Text Length (Word Count) x100')
    axs[i].set_ylabel('Frequency x100')
    axs[i].set_title(f'{category} Distribution')

# Show the plot
plt.tight_layout()
plt.show()