import gensim
import pandas as pd
import numpy as np
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


# Tokenize and preprocess tweets
english_words = set(words.words())
stop_words = set(stopwords.words('english'))
punctuation = set(string.punctuation)

def preprocess_tweet(tweet):
    tokens = word_tokenize(tweet)
    tokens = [word.lower() for word in tokens if word.isalpha()]  # Remove non-alphabetic tokens
    tokens = [word for word in tokens if word not in stop_words and word not in punctuation and word in english_words]  # Remove stopwords and punctuation
    return tokens

sentence = 'Democracy, Republic, Monarchy, Autocracy, Oligarchy, Dictatorship, Totalitarianism, Anarchy, Government, Constitution, Parliament, Congress, Senate, Legislature, Judiciary, Executive, President, Prime Minister, Governor, Mayor, Minister, Election, Campaign, Vote, Ballot, Candidate, Party, Platform, Debate, Policy, Law, Bill, Amendment, Referendum, Plebiscite, Suffrage, Citizenship, Nationalism, Federalism, State, Municipality, Bureaucracy, Regulation, Taxation, Budget, Welfare, Subsidy, Sanction, Diplomacy, Treaty, Alliance, Conflict, War, Peace, Negotiation, Mediation, Arbitration, Resolution, Protest, Movement, Rights, Freedom, Liberty, Justice, Equality, Equity, Diversity, Inclusion, Representation, Constituency, District, Quorum, Caucus, Majority, Minority, Coalition, Opposition, Ideology, Conservatism, Liberalism, Socialism, Communism, Capitalism, Fascism, Nationalism, Populism, Globalization, Secularism, Theocracy, Pluralism, Patriotism, Sovereignty, Autonomy, Annexation, Colonization, Imperialism, Revolution, Coup, Insurgency, Terrorism, Extremism, Radicalism, Moderation, Diplomat, Ambassador, Envoy, Consul, Attaché, Intelligence, Espionage, Surveillance, Humanitarian, NGO, Civil Society, Activism, Lobbying, Advocacy, Petition, Boycott, Strike, Rally, March, Demonstration, Riot, Insurrection, Rebellion, Guerrilla, Militia, Conflict, Warfare, Arms, Military, Soldier, Army, Navy, Air Force, Marine, Commander, General, Admiral, Veteran, Casualty, Truce, Ceasefire, Armistice, Disarmament, Nonproliferation, Treaty, Summit, Conference, Convention, Protocol, Accord, Agreement, Compact, Pact, Negotiator, Mediator, Arbitrator, Facilitator, Peacemaker, Broker, Delegate, Representative, Diplomatic, Embassy, Consulate, Mission, Visa, Passport, Immunity, Extradition, Asylum, Refugee, Migrant, Border, Territory, Region, Province, County, City, Town, Rural, Urban, Metropolitan, Capital, Municipality, District, Zone, Area, Locale, Community, Society, Population, Demographics, Census, Quota, Minority, Majority, Group'
liststr = preprocess_tweet(sentence)
print(liststr)