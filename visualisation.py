import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
import numpy as np


def plot_data(df, class_name):
    ax = df[class_name].value_counts().plot(kind='bar',
                                                    figsize=(12, 8),
                                                    title="Class Distribution")
    total = 0
    for p in ax.patches:
        total += p.get_height()
    for p in ax.patches:
        ax.annotate(np.round(p.get_height(), decimals=2), (p.get_x() + p.get_width() / 2, p.get_height() / 2),
                    ha='center', va='center', xytext=(0, 10), textcoords='offset points',
                    bbox=dict(facecolor='yellow', alpha=0.5))

        ax.annotate(str(np.round(p.get_height() / total * 100, decimals=2)) + "%",
                    (p.get_x() + p.get_width() / 2, p.get_height() / 2),
                    ha='center', va='center', xytext=(0, -20), textcoords='offset points',
                    bbox=dict(facecolor='gray', alpha=0.5))


def plot_word_cloud(df):
    text = df.text.values
    wordcloud = WordCloud(
        width = 3000,
        height = 2000,
        background_color = 'black',
        stopwords = STOPWORDS).generate(str(text))
    fig = plt.figure(
        figsize = (40, 30),
        facecolor = 'k',
        edgecolor = 'k')
    plt.imshow(wordcloud, interpolation = 'bilinear')
    plt.axis('off')
    plt.tight_layout(pad=0)
    plt.show()