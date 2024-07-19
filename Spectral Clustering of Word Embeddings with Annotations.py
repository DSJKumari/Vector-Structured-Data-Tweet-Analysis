import gensim
import pandas as pd
import numpy as np
import seaborn as sns
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords, words
from gensim.models import KeyedVectors
from nltk.data import find
import string
import nltk
from sklearn.decomposition import PCA
from sklearn.cluster import SpectralClustering
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from bokeh.plotting import figure, show, output_notebook, output_file
import bokeh.plotting as bp
from bokeh.models import HoverTool, BoxSelectTool, ColumnDataSource, CategoricalColorMapper
from bokeh.palettes import Category10 

file1_path = '/Users/jharna/Documents/ALGORITHMS&DATA_MODELS/GP2/russian-troll-tweets-master/IRAhandle_tweets_1.csv'
file2_path = '/Users/jharna/Documents/ALGORITHMS&DATA_MODELS/GP2/russian-troll-tweets-master/IRAhandle_tweets_2.csv'
file3_path = '/Users/jharna/Documents/ALGORITHMS&DATA_MODELS/GP2/russian-troll-tweets-master/IRAhandle_tweets_3.csv'
file4_path = '/Users/jharna/Documents/ALGORITHMS&DATA_MODELS/GP2/russian-troll-tweets-master/IRAhandle_tweets_4.csv'
file5_path = '/Users/jharna/Documents/ALGORITHMS&DATA_MODELS/GP2/russian-troll-tweets-master/IRAhandle_tweets_5.csv'
file6_path = '/Users/jharna/Documents/ALGORITHMS&DATA_MODELS/GP2/russian-troll-tweets-master/IRAhandle_tweets_6.csv'
file7_path = '/Users/jharna/Documents/ALGORITHMS&DATA_MODELS/GP2/russian-troll-tweets-master/IRAhandle_tweets_7.csv'
file8_path = '/Users/jharna/Documents/ALGORITHMS&DATA_MODELS/GP2/russian-troll-tweets-master/IRAhandle_tweets_8.csv'
file9_path = '/Users/jharna/Documents/ALGORITHMS&DATA_MODELS/GP2/russian-troll-tweets-master/IRAhandle_tweets_9.csv'
file10_path = '/Users/jharna/Documents/ALGORITHMS&DATA_MODELS/GP2/russian-troll-tweets-master/IRAhandle_tweets_10.csv'
file11_path = '/Users/jharna/Documents/ALGORITHMS&DATA_MODELS/GP2/russian-troll-tweets-master/IRAhandle_tweets_11.csv'
file12_path = '/Users/jharna/Documents/ALGORITHMS&DATA_MODELS/GP2/russian-troll-tweets-master/IRAhandle_tweets_12.csv'
file13_path = '/Users/jharna/Documents/ALGORITHMS&DATA_MODELS/GP2/russian-troll-tweets-master/IRAhandle_tweets_13.csv'

df1 = pd.read_csv(file1_path)
df2 = pd.read_csv(file2_path)
df3 = pd.read_csv(file3_path)
df4 = pd.read_csv(file4_path)
df5 = pd.read_csv(file5_path)
df6 = pd.read_csv(file6_path)
df7 = pd.read_csv(file7_path)
df8 = pd.read_csv(file8_path)
df9 = pd.read_csv(file9_path)
df10 = pd.read_csv(file10_path)
df11 = pd.read_csv(file11_path)
df12 = pd.read_csv(file12_path)
df13 = pd.read_csv(file13_path)

combined_df = pd.concat([df1, df2, df3, df4, df5, df6, df7, df8, df9, df10, df11, df12, df13], ignore_index=True)
combined_df['content'] = combined_df['content'].fillna('')

english_words = set(words.words())
stop_words = set(stopwords.words('english'))
punctuation = set(string.punctuation)

def preprocess_tweet(tweet):
    tokens = word_tokenize(tweet)
    tokens = [word.lower() for word in tokens if word.isalpha()]  # Remove non-alphabetic tokens
    tokens = [word for word in tokens if word not in stop_words and word not in punctuation]  # Remove stopwords and punctuation
    return tokens

combined_df['tokens'] = combined_df['content'].apply(preprocess_tweet)

model = Word2Vec(sentences=combined_df['tokens'], vector_size=100, window=5, min_count=1, workers=4)

model.save('word2vec_model.model')

loaded_model = Word2Vec.load('word2vec_model.model')

word2vec_sample = str(find('models/word2vec_sample/pruned.word2vec.txt'))
pretrained_model = gensim.models.KeyedVectors.load_word2vec_format(word2vec_sample, binary=False)

sentence = 'Trump, Donald, Hillary, Clinton, Democracy, Republic, Monarchy, Autocracy, Oligarchy, Dictatorship, Totalitarianism, Anarchy, Government, Constitution, Parliament, Congress, Senate, Legislature, Judiciary, Executive, President, Prime Minister, Governor, Mayor, Minister, Election, Campaign, Vote, Ballot, Candidate, Party, Platform, Debate, Policy, Law, Bill, Amendment, Referendum, Plebiscite, Suffrage, Citizenship, Nationalism, Federalism, State, Municipality, Bureaucracy, Regulation, Taxation, Budget, Welfare, Subsidy, Sanction, Diplomacy, Treaty, Alliance, Conflict, War, Peace, Negotiation, Mediation, Arbitration, Resolution, Protest, Movement, Rights, Freedom, Liberty, Justice, Equality, Equity, Diversity, Inclusion, Representation, Constituency, District, Quorum, Caucus, Majority, Minority, Coalition, Opposition, Ideology, Conservatism, Liberalism, Socialism, Communism, Capitalism, Fascism, Nationalism, Populism, Globalization, Secularism, Theocracy, Pluralism, Patriotism, Sovereignty, Autonomy, Annexation, Colonization, Imperialism, Revolution, Coup, Insurgency, Terrorism, Extremism, Radicalism, Moderation, Diplomat, Ambassador, Envoy, Consul, Attaché, Intelligence, Espionage, Surveillance, Humanitarian, NGO, Civil Society, Activism, Lobbying, Advocacy, Petition, Boycott, Strike, Rally, March, Demonstration, Riot, Insurrection, Rebellion, Guerrilla, Militia, Conflict, Warfare, Arms, Military, Soldier, Army, Navy, Air Force, Marine, Commander, General, Admiral, Veteran, Casualty, Truce, Ceasefire, Armistice, Disarmament, Nonproliferation, Treaty, Summit, Conference, Convention, Protocol, Accord, Agreement, Compact, Pact, Negotiator, Mediator, Arbitrator, Facilitator, Peacemaker, Broker, Delegate, Representative, Diplomatic, Embassy, Consulate, Mission, Visa, Passport, Immunity, Extradition, Asylum, Refugee, Migrant, Border, Territory, Region, Province, County, City, Town, Rural, Urban, Metropolitan, Capital, Municipality, District, Zone, Area, Locale, Community, Society, Population, Demographics, Census, Quota, Minority, Majority, Group'
words = preprocess_tweet(sentence)

embeddings = [model.wv[word] for word in words]
embeddings_array = np.array(embeddings)
similarity_matrix = cosine_similarity(embeddings_array)
n_samples = embeddings_array.shape[0]
perplexity_value = min(30, n_samples - 1) 
print('Clustering start')

clustering = SpectralClustering(n_clusters=5, affinity='precomputed')
labels = clustering.fit_predict(similarity_matrix)
print('Before Transform')

tsne_model = TSNE(n_components=2, perplexity=perplexity_value, random_state=42)
reduced_embeddings = tsne_model.fit_transform(embeddings_array)

tsne_df = pd.DataFrame(reduced_embeddings, columns=['x', 'y'])
tsne_df['words'] = words
tsne_df['labels'] = labels
print('Before plot')

source = ColumnDataSource(tsne_df)

output_file("tsne_plot.html", title="t-SNE of Word Vectors")

plotf = figure(plot_width=800, plot_height=800, title="t-SNE of Word Vectors", tools="pan,wheel_zoom,box_zoom,save,reset")

plotf.scatter('x', 'y', size=10, source=source)

hover = HoverTool()
hover.tooltips = [("word", "@words")]
plotf.add_tools(hover)
show(plotf)

import matplotlib.pyplot as plt

plt.figure(figsize=(15, 12))

# Scatter plot for each cluster
for i in range(5):
    cluster_indices = labels == i
    plt.scatter(reduced_embeddings[cluster_indices, 0], reduced_embeddings[cluster_indices, 1], label=f'Cluster {i}')

    # Annotate each point in the cluster
    for idx in np.where(cluster_indices)[0]:
        plt.annotate(words[idx],
                     (reduced_embeddings[idx, 0], reduced_embeddings[idx, 1]),
                     textcoords="offset points",
                     xytext=(5,5),
                     ha='right')

plt.legend()
plt.title('Spectral Clustering of Word Embeddings with Annotations')
plt.xlabel('')
plt.ylabel('')
plt.show()