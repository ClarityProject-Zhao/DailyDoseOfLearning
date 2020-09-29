import numpy as np
import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
import importlib
import sklearn

folder = os.getcwd()
data_2020 = pd.read_csv(os.path.join(folder, r'data/2020.csv'))

##to do:
# ad PCA and LDA

dict_region = {'Western Europe': 'W.Eur',
               'North America and ANZ': 'NA,ANZ',
               'Middle East and North Africa': "Mid.East, N.Africa",
               'Latin America and Caribbean': 'SA',
               'Central and Eastern Europe': 'Cen.E,EUR',
               'East Asia': 'E.Asia',
               'Southeast Asia': 'S.Asia',
               'Commonwealth of Independent States': 'Ind.States',
               'Sub-Saharan Africa': 'SS.Africa',
               'South Asia': 'S.Asia'
               }
data_2020['region'] = data_2020['Regional indicator'].apply(lambda x: dict_region[x])
# distribution plot
# bar plot
plt.rcParams.update({'figure.autolayout': True})
ax_bar = sns.barplot(data=data_2020, x='region', y='Ladder score')  # axe level object
ax_bar.set_xticklabels(labels=ax_bar.get_xticklabels(), rotation=40, horizontalalignment='right')

# violin plot
plt.rcParams.update({'figure.autolayout': True})
fig_violin = sns.catplot(data=data_2020, x='region', y='Ladder score', kind='violin')
fig_violin.set_xticklabels(fig_violin.axes[0, 0].get_xticklabels(), rotation=40, horizontalalignment='right')

# displot by region
plt.rcParams.update({'figure.autolayout': True})
fig_dis = sns.displot(data=data_2020, x="Ladder score", col="region", legend=False)
fig_dis.set(xlabel=None)

# pair plot
data_score = data_2020[['Ladder score', 'Logged GDP per capita', 'Social support', 'Healthy life expectancy',
                        'Freedom to make life choices', 'Generosity', 'Perceptions of corruption']]
sns.pairplot(data_score)

g = sns.PairGrid(data_score)
g.map_upper(sns.regplot)
g.map_lower(sns.kdeplot)
g.map_diag(sns.histplot, kde=True)

g1 = sns.lmplot(data=data_2020, x="Logged GDP per capita", y="Ladder score", col='region')
g1.set(xlabel=None)
g2 = sns.lmplot(data=data_2020, x="Social support", y="Ladder score", col='region')
g2.set(xlabel=None)
g3 = sns.lmplot(data=data_2020, x="Healthy life expectancy", y="Ladder score", col='region')
g3.set(xlabel=None)
g4 = sns.lmplot(data=data_2020, x="Freedom to make life choices", y="Ladder score", col='region')
g4.set(xlabel=None)
g5 = sns.lmplot(data=data_2020, x="Generosity", y="Ladder score", col='region')
g5.set(xlabel=None)
g6 = sns.lmplot(data=data_2020, x="Perceptions of corruption", y="Ladder score", col='region')
g6.set(xlabel=None)

from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.metrics.pairwise import pairwise_distances_argmin

data_x = data_2020[['Logged GDP per capita', 'Social support', 'Healthy life expectancy',
                    'Freedom to make life choices', 'Generosity', 'Perceptions of corruption']]
# need to decide n_clusters
# Elbow method
WSS = {}
for i in range(2, 30):
    k_means = KMeans(init='k-means++', n_clusters=i, n_init=10)
    k_means.fit(data_x)
    k_means_labels, k_means_distance = metrics.pairwise_distances_argmin_min(data_x, k_means.cluster_centers_)
    WSS[i] = np.sum(np.sqrt(k_means_distance))
WSS_pd = pd.DataFrame(WSS.values(), index=WSS.keys(), columns=['WSS'])
WSS_pd.plot()
# no clean answer
from sklearn.metrics import silhouette_score

sil = {}
for i in range(2, 30):
    k_means = KMeans(init='k-means++', n_clusters=i, n_init=10)
    k_means.fit(data_x)
    k_means_labels, k_means_distance = metrics.pairwise_distances_argmin_min(data_x, k_means.cluster_centers_)
    sil[i] = silhouette_score(data_x, k_means_labels, metric='euclidean')
sil_pd = pd.DataFrame(sil.values(), index=sil.keys(), columns=['silhouette'])
sil_pd.plot()
# 2 is the winner

# try dimension reduction
# include dimensionality reduction
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
data_x_pca = pca.fit_transform(data_x)

sil_pca = {}
for i in range(2, 30):
    k_means = KMeans(init='k-means++', n_clusters=i, n_init=10)
    k_means.fit(data_x_pca)
    k_means_labels, k_means_distance = metrics.pairwise_distances_argmin_min(data_x_pca, k_means.cluster_centers_)
    sil_pca[i] = silhouette_score(data_x_pca, k_means_labels, metric='euclidean')
sil_pca_pd = pd.DataFrame(sil_pca.values(), index=sil_pca.keys(), columns=['silhouette'])
sil_pca_pd.plot()
# 2 is still the winner


k_means = KMeans(init='k-means++', n_clusters=2, n_init=10)
k_means.fit(data_x)
k_means_labels, k_means_distance = metrics.pairwise_distances_argmin_min(data_x, k_means.cluster_centers_)

data_2020.loc[:, 'KN_label'] = pd.Series(k_means_labels)
data_2020.loc[:, 'KN_distance'] = pd.Series(k_means_distance)
data_2020[['Country name', 'KN_label']]

# plot cluster in 2D surface
data_pca_results = pd.DataFrame(data_x_pca, columns=['x1', 'x2']).join(pd.DataFrame(k_means_labels, columns=['label']))
sns.jointplot(data=data_pca_results, x='x1', y='x2', hue='label').plot_joint(sns.kdeplot)

# loadings
loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
loadings_pd = pd.DataFrame(loadings.T, columns=data_x.columns.tolist())
loadings_pd.plot(kind='bar')

# plot dist within cluster
sns.displot(data=data_2020, x='KN_distance', col='KN_label')

g1 = sns.lmplot(data=data_2020, x="Logged GDP per capita", y="Ladder score", col='KN_label')
g1.set(xlabel=None)
g2 = sns.lmplot(data=data_2020, x="Social support", y="Ladder score", col='KN_label')
g2.set(xlabel=None)
g3 = sns.lmplot(data=data_2020, x="Healthy life expectancy", y="Ladder score", col='KN_label')
g3.set(xlabel=None)
g4 = sns.lmplot(data=data_2020, x="Freedom to make life choices", y="Ladder score", col='KN_label')
g4.set(xlabel=None)
g5 = sns.lmplot(data=data_2020, x="Generosity", y="Ladder score", col='KN_label')
g5.set(xlabel=None)
g6 = sns.lmplot(data=data_2020, x="Perceptions of corruption", y="Ladder score", col='KN_label')
g6.set(xlabel=None)

data = {}
data_all = pd.DataFrame()
for year in range(2015, 2021):
    data[year] = pd.read_csv(os.path.join(folder, r'data/', f'{str(year)}.csv'))
    data[year].rename(columns={'Country name': 'Country', 'Country or region': 'Country'}, inplace=True)
    data[year].rename(columns={'Ladder score': f'{str(year)}', 'Happiness Score': f'{str(year)}',
                               'Happiness.Score': f'{str(year)}', 'Score': f'{str(year)}'},
                      inplace=True)
    data_all = pd.merge(data_all, data[year][['Country', f'{str(year)}']], on='Country') if not data_all.empty else \
    data[year][['Country', f'{str(year)}']]

data_combo = pd.merge(data_all, data_2020[['Country name', 'region', 'KN_label']], left_on='Country',
                      right_on='Country name')
data_combo.drop(columns='Country name', inplace=True)
data_combo_long = pd.melt(data_combo, id_vars=['Country', 'region', 'KN_label'],
                          value_vars=[f'{str(x)}' for x in range(2015, 2021)])
data_combo_long.columns = ['Country', 'Region', 'KN_label', 'Year', 'Score']
sns.relplot(data=data_combo_long, x='Year', y='Score', kind='line')
sns.relplot(data=data_combo_long, x='Year', y='Score', col='KN_label', kind='line')
sns.relplot(data=data_combo_long, x='Year', y='Score', col='Region', col_wrap=2, hue='Country', legend=False)

#plot countries within region
fig = plt.figure(figsize=(20, 40))
plt.tight_layout()
axes = fig.add_gridspec(5, 2)

for i, region_str in enumerate(dict_region.values()):
    ax=fig.add_subplot(axes[i // 2, i % 2])
    sns.pointplot(data=data_combo_long.loc[data_combo_long['Region'] == region_str, :], x='Year', y='Score',
                  col='Region', col_wrap=2, hue='Country', ax=ax)
    ax.legend(loc='lower right',ncol=2,framealpha=0.3)
    ax.set_ylim(data_combo_long['Score'].min(),data_combo_long['Score'].max())