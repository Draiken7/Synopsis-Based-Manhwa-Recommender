# Synopsis-Based-Manhwa-Recommender
A simple [Manhwa](https://en.wikipedia.org/wiki/Manhwa) Recommender based on a [Kaggle](https://www.kaggle.com/datasets/iridazzle/webtoon-originals-datasets?select=webtoon_originals_en.csv) dataset.
Sources and References:
1. [AnalyticsVidhya](https://www.analyticsvidhya.com/blog/2021/07/recommendation-system-understanding-the-basic-concepts/)
2. [Medium](https://medium.com/@hazallgultekin/what-is-silhouette-score-f428fb39bf9a)
3. [geeksforgeeks](https://www.geeksforgeeks.org/davies-bouldin-index/)

The recommnder is based on dataset of manhwa comics from kaggle. It uses [BERT](https://huggingface.co/docs/transformers/en/model_doc/bert) model to generate embeddings and uses averaged representations. The representation(768 dimensional vectors) are then used for feature selection using [PCA](https://www.ibm.com/think/topics/principal-component-analysis). The idea was to generate latent sub genre categories based on the synopsis embeddings to cluster the titles.


# Procedure
## 1. Preprocessing and BERT
The data for synopsis based recommender only uses synopsis from the iven dataset along with titles. All duplicates and titles with 0 synopsis length or missing values were dropped. The number of samples in this processed dataset is 2688.

The HuggingFace implementation of BERT (`google-bert/bert-base-uncased"`) was used (both for tokenizer and model) to generate embeddings for the synopsis text. Truncation of text due to large sequence length only affects 1 sample. The tensor was then averaged across sequence length to generate a single averaged tensor for the entire synopsis embeddings. This step hopes to capture the essence of the synopsis in a single vector and hence use this representation of the crux of the synopsis to do clustering of similar such vectors.


## 2. Feature Selection, Models and Metrics
### Feature Selection
Pricipal Component Analysis was used to perform feature selection on the 768 dimensional data. The motivation was that reducing number of features may help achieve better scores for various models. Cross referencing various values for number of pricipal components agains the metrics used for the project (Silhouette and Davies Bouldin Score) also suggests lower dimensional data generates better performance for the models.


### Models
Scikit documentation lists all available clustering models ([Here](https://scikit-learn.org/stable/modules/clustering#hierarchical-clustering)) along with thier use cases and other important details such as parameters and scalability. Based on these and some test runs of various models (including DBSCAN, HDBSCAN, OPTICS and BIRCH), I chose the following four models to work with for this project:

- Affinity Propagation: Works for non falat geometry and works with many clusters with varying sizes. I used it give a good estimate of number of clusters.
- KMeans: The generic kmeans was used to provide a baseline for other models
- Agglomerative Clustering: Selected for similar reasons as that for Affinity Propagation and generated better results during the testing phase
- Gaussian Mixture: I selected Gaussian Mixture models as they give good densit estimations and since it generated best results during the initial testing phase.


### Performance Metrics
Finding good perfromace indicating metrics for unsupervised learning algorithms was a challenge and the two metrics that seemed logical for this project were [Silhouette](https://medium.com/@hazallgultekin/what-is-silhouette-score-f428fb39bf9a) and [Davies Bouldin](https://www.geeksforgeeks.org/davies-bouldin-index/) Score. Most other metrics rely on some form of ground truth whereas these metrics dont and generate an easy to understand value as well. I aimed to choose a model with a good silhouette and DAvies bouldin score. Informed reasonging helped me infer that the silhouette score could not be 1, as cluster may be overlapping and vectors within clusters spread apart. Similarly, the Davies bouldin score could not be 0 since i expected the intra cluster distance vs inter cluster distance to be larger signifying  theat the clusters are close together and large. Both these notions are backed by the idea that a single manhwa title may belong to multiple latent sub genres (clusters). The aim was to get a high positive result for silhouette score and a non negative lowest score for DAvies bouldin score.

First step was to check if there is an optimal range of features for which the selected models perform better. The number of principal components started at 3 to almost 256. the following are the results:
- **KMEANS**
  - ![image](https://github.com/user-attachments/assets/6af5c62a-ffd6-4cf1-a738-81adc57d0c41)
    For Kmeans, the silhouette score tends to converge closer to zero with increasing number of clusters as well as for increasing number of principal components. Note that this trend is only broken when number of principal components is 3, which results in highest score (closer to ideal 1) for all clustering sizes.
  - ![image](https://github.com/user-attachments/assets/0d13ae07-0d23-4887-8626-a65244350e47)
    The davies bouldin score also shows reducing values with increasing cluster sizes but the trend is not clear for increasing principal components. A clear minimum for all clustering sizes is visible for 3 principal components

- **Agglomerative Clustering**
  - 
