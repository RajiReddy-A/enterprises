import pandas as pd
from sklearn.cluster import KMeans
#from gensim.Models import Word2Vec

path = 'newdata(NNP_NN_NS_VB_ADV).xlsx' # set reviews file path.

data = pd.read_excel(path)
data.head(n=2)

data["EnterpriseName"] = data["EnterpriseName"].astype(str)
extracted_names = data["EnterpriseName"]
extracted_names
from sklearn.feature_extraction.text import TfidfVectorizer
# vectorization of the texts
vectorizer = TfidfVectorizer(min_df=0.001)
X = vectorizer.fit_transform(extracted_names)
document_term_matrix = pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names())    # DTM to Dataframe.

document_term_matrix.shape

document_term_matrix.head(10)

data['Age'] = (pd.to_datetime('now') - data['CommmenceDate']).dt.days

all_cols = ['AID', 'EnterpriseName', 'SocialCategory', 'Gender', 'PH',
       'OrganisationType', 'PlantLocation', 'Address', 'State', 'District',
       'PINCode', 'CommmenceDate', 'MajorActivity', 'EnterpriseType',
       'NIC5DigitCode', 'TotalEmp', 'InvestmentCost', 'Dic_Name',
       'RegistrationDate', 'LG_Dist_Code', 'Age']

req_cols = ["District", "Age", "MajorActivity", "EnterpriseType", "TotalEmp", "InvestmentCost"]
new_data = data[req_cols]

new_data = pd.get_dummies(new_data)
new_data.shape
document_term_matrix.shape

final_df = pd.concat([new_data.reset_index(drop=True), document_term_matrix], axis=1)
final_df.shape

# Number of clusters
kmeans = KMeans(n_clusters=50)
# Fitting the input data
kmeans = kmeans.fit(final_df)
# Getting the cluster labels

groups = kmeans.predict(final_df)
# Centroid values
centroids = kmeans.cluster_centers_
centroids.shape
print(centroids) # From sci-kit learn
print(len(groups))
print(groups[:5])
print(groups)

df = pd.DataFrame(groups)
df.columns = ['Groups']

from collections import Counter
a = dict(Counter(groups))
a
"""
import sister
embedder = sister.MeanEmbedding(lang="en")

sentence = "this is cat."
vector = embedder(sentence)
"""

