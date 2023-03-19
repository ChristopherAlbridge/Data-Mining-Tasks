import pandas as pd
from pandas.api.types import CategoricalDtype
from wordcloud import WordCloud
from matplotlib import pyplot as plt


fake_news = pd.read_csv('https://raw.githubusercontent.com/HAN-M3DM-Data-Mining/assignments/master/datasets/NB-fakenews.csv')
fake_news.head()

catID = CategoricalDtype(categories = [0,1], ordered = True)

fake_news.label.value_counts()

fake_news.label.value_counts(normalize = False)

wc_untrue = WordCloud(background_color='white', colormap='Blues').generate(untrue_text)

wc_real = WordCloud(background_color='white', colormap='Reds').generate(real_text)


fig, (wc1, wc2) = plt.subplots(1,2)
fig.suptitle('Wordclouds for untrue and real')
wc1.imshow(wc_untrue)
wc2.imshow(wc_real)
plt.show()

#Vectors
print(fake_news['text'].isna().sum())
fake_news.dropna(subset=['text'], inplace=True)

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix


vectorizer = TfidfVectorizer(max_features=1000)
vectors = vectorizer.fit_transform(fake_news.text)
wordsfake_news = pd.DataFrame(vectors.toarray(), columns=vectorizer.get_feature_names_out())
wordsfake_news.head()


xTrain, xTest, yTrain, yTest = train_test_split(wordsfake_news, fake_news.label)

#Testing the data

bayes = MultinomialNB()
bayes.fit(xTrain, yTrain)

yPred = bayes.predict(xTest)
yTrue = yTest

accuracyScore = accuracy_score(yTrue, yPred)
print(f'Accuracy: {accuracyScore}')

matrix = confusion_matrix(yTrue, yPred)
labelNames = pd.Series([0, 1])
pd.DataFrame
