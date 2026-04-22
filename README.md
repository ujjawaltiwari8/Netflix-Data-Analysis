# Netflix-Data-Analysis
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from scipy import stats
from scipy.stats import shapiro, chi2_contingency, norm

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


print("=" * 55)
print("UNIT I – Loading the Dataset")
print("=" * 55)

df = pd.read_csv("C:/Users/rishu/Downloads/netflix_titles.csv")

print("Shape of dataset:", df.shape)
print("\nColumn names:", df.columns.tolist())
print("\nFirst 5 rows:")
print(df.head())
print("\nData types:")
print(df.dtypes)


print("\n" + "=" * 55)
print("UNIT II – Data Cleaning & Manipulation")
print("=" * 55)

print("\n'type' column distribution:")
print(df['type'].value_counts())

print("\nMissing values before cleaning:")
print(df.isnull().sum())

df['director'] = df['director'].fillna('Unknown')
df['cast'] = df['cast'].fillna('Unknown')
df['country'] = df['country'].fillna('Unknown')
df['rating'] = df['rating'].fillna('Unknown')
df['duration'] = df['duration'].fillna('Unknown')
df['date_added'] = df['date_added'].fillna('Unknown')

print("\nMissing values after cleaning:")
print(df.isnull().sum())

df['date_added_clean'] = pd.to_datetime(
    df['date_added'].str.strip(),
    format='%B %d, %Y',
    errors='coerce'
)

df['year_added'] = df['date_added_clean'].dt.year

df['duration_min'] = df['duration'].str.extract(r'(\d+)').astype(float)

df['primary_country'] = df['country'].apply(lambda x: str(x).split(',')[0].strip())
df['primary_genre'] = df['listed_in'].apply(lambda x: str(x).split(',')[0].strip())

movie_dur = df[df['type'] == 'Movie']['duration_min'].dropna().values

print("\nMovie Duration Stats:")
print("Mean:", np.mean(movie_dur))
print("Median:", np.median(movie_dur))
print("Std:", np.std(movie_dur))


print("\n" + "=" * 55)
print("UNIT III – Data Visualization")
print("=" * 55)

sns.set_theme(style='whitegrid')

type_counts = df['type'].value_counts()

plt.figure(figsize=(6, 6))
plt.pie(type_counts, labels=type_counts.index, autopct='%1.1f%%')
plt.title('Movies vs TV Shows')
plt.savefig('pie.png')
plt.show()


yearly = df.groupby(['year_added', 'type']).size().unstack(fill_value=0)
yearly = yearly[(yearly.index >= 2010) & (yearly.index <= 2021)]

plt.figure(figsize=(10, 5))
plt.plot(yearly.index, yearly['Movie'], marker='o')
plt.plot(yearly.index, yearly['TV Show'], marker='s')
plt.legend(['Movie', 'TV Show'])
plt.title('Yearly Trend')
plt.savefig('trend.png')
plt.show()


top_countries = df['primary_country'].value_counts().head(10)

plt.figure(figsize=(10, 5))
sns.barplot(x=top_countries.values, y=top_countries.index)
plt.title('Top Countries')
plt.savefig('countries.png')
plt.show()


print("\n" + "=" * 55)
print("UNIT IV – EDA")
print("=" * 55)

print(df[['release_year', 'duration_min']].describe())

df['type_num'] = (df['type'] == 'Movie').astype(int)

corr_data = df[['release_year', 'duration_min', 'type_num', 'year_added']].dropna()

print("\nCorrelation:")
print(corr_data.corr())

Q1 = np.percentile(movie_dur, 25)
Q3 = np.percentile(movie_dur, 75)
IQR = Q3 - Q1

lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR

outliers = df[(df['duration_min'] < lower) | (df['duration_min'] > upper)]

print("Outliers:", len(outliers))

plt.boxplot(movie_dur)
plt.title("Boxplot")
plt.savefig('box.png')
plt.show()


print("\n" + "=" * 55)
print("UNIT V – Statistical Analysis")
print("=" * 55)

movies_yr = df[df['type'] == 'Movie']['release_year'].dropna()
tv_yr = df[df['type'] == 'TV Show']['release_year'].dropna()

n1, n2 = len(movies_yr), len(tv_yr)

z = (movies_yr.mean() - tv_yr.mean()) / np.sqrt(
    (movies_yr.std()**2 / n1) + (tv_yr.std()**2 / n2)
)

p = 2 * (1 - norm.cdf(abs(z)))

print("Z-test:", z, p)

t, p2 = stats.ttest_ind(movies_yr, tv_yr)
print("T-test:", t, p2)

sample = pd.Series(movie_dur).sample(500, random_state=42)
sw, p3 = shapiro(sample)
print("Shapiro:", sw, p3)


print("\n" + "=" * 55)
print("ML – Logistic Regression")
print("=" * 55)

valid_ratings = [
    'G','PG','PG-13','R','TV-Y','TV-Y7','TV-G','TV-PG','TV-14','TV-MA'
]

ml_df = df[['type','release_year','rating','primary_genre']]
ml_df = ml_df[ml_df['rating'].isin(valid_ratings)].dropna()

le1 = LabelEncoder()
le2 = LabelEncoder()

ml_df['rating_enc'] = le1.fit_transform(ml_df['rating'])
ml_df['genre_enc'] = le2.fit_transform(ml_df['primary_genre'])

X = ml_df[['release_year','rating_enc','genre_enc']]
y = (ml_df['type'] == 'Movie').astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)

sns.heatmap(cm, annot=True, fmt='d')
plt.title("Confusion Matrix")
plt.savefig('cm.png')
plt.show()
