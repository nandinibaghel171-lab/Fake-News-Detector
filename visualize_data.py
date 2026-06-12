import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from wordcloud import WordCloud

# Data load karo
df = pd.read_csv('data/fake.csv')

# String convert karo
df['title'] = df['title'].fillna('').astype(str)
df['text'] = df['text'].fillna('').astype(str)

# Graph 1: Title length
df = df[df['title'] != 'nan']
df['title_length'] = df['title'].apply(lambda x: len(str(x)))
print(df['title_length'].describe())
df = df.dropna(subset=['title_length'])
plt.figure(figsize=(10, 5))
plt.hist(df['title_length'], bins=50, color='red')
plt.title('Fake News Title Length')
plt.xlabel('Length')
plt.ylabel('Count')
plt.show()

# Graph 2: Top 10 words
words = ' '.join(df['title'].astype(str)).split()
word_count = Counter(words).most_common(10)
words_df = pd.DataFrame(word_count, 
            columns=['Word', 'Count'])
plt.figure(figsize=(10, 5))
sns.barplot(x='Count', y='Word', data=words_df)
plt.title('Top 10 Words in Fake News')
plt.show()

# Graph 3: Word Cloud
text = ' '.join(df['title'])
wc = WordCloud(width=800, height=400,
        background_color='black').generate(text)
plt.figure(figsize=(10, 5))
plt.imshow(wc)
plt.axis('off')
plt.title('Fake News Word Cloud')
plt.show()