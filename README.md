![image](https://github.com/user-attachments/assets/fcdc85c0-8934-4667-b4ec-d72ab61cd5ce)# Synopsis-Based-Manhwa-Recommender
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
(**NOTE** PCA 0 indicates all features are taken as is.)

- **Silhouette Score vs Number of Principal Components vs Number of Clusters**
  There seems to be a general trend for Kmeans, Agglomerative clustering and Gaussain Mixture models that as the number of clusters increase, the score drops. Similarly as the number of principal components increases, there is a somewhat decreasing trend for the scores. A noticable spike happens when the number of principal components is set to 3.
  
  - **KMeans**  
    ![image](https://github.com/user-attachments/assets/6af5c62a-ffd6-4cf1-a738-81adc57d0c41)
  
  - **Agglomerative Clustering**
    ![image](https://github.com/user-attachments/assets/7baf8e59-e873-4ec9-8cc1-3cea9afe58e4)

  - **Gaussian Mixture**
    ![image](https://github.com/user-attachments/assets/3d44105e-5912-48b0-83d0-7021a1d7cf43)

  - **Affinity Propagation**
    
    ![image](https://github.com/user-attachments/assets/e00db861-9c09-4428-9d20-3e14b74030f1)
    ![image](https://github.com/user-attachments/assets/a5b4c991-afa2-4ca9-970f-ff40f69bd919)
    ![image](https://github.com/user-attachments/assets/a88c93e6-42e8-4bdc-96f5-1a833c0d1ad8)

- **Davies-Bouldin Score vs Number of Principal Components vs Number of Clusters**
  The scores seem to decrease with increasing number of clusters and increase with increasing number of principal components. The minimum score can be observed here when the number of princcipal components is 3.

  - **KMeans**
    ![image](https://github.com/user-attachments/assets/f22aaeba-214d-441a-8b02-1f03e3237300)

  - **Agglomerative Clustering**   ![image](https://github.com/user-attachments/assets/89ecb139-b9ee-4ae2-b736-b210f5ad8a71)

  - **Gaussian Mixture**   ![image](https://github.com/user-attachments/assets/93ae5ed8-9f82-49d2-b525-9925546c7cbd)
 
  - **Affinity Propagation**
 
    ![image](https://github.com/user-attachments/assets/86318211-fc35-4f5f-8dd1-2fe86b7176f9)
    ![image](https://github.com/user-attachments/assets/2ade666c-1f54-4496-b4c8-687c54bb2ccd)
    ![image](https://github.com/user-attachments/assets/3486206c-628d-49af-ab72-7423ecbba6ea)


## 3. Hper Parameter Tuning
### Number of Principal Components
Various values for PCA were tested against 18 clusters for Kmeans, Agglomerative Clustering and Gaussain Mixture models. The dataset originally has 18 disticnt `genre` tags which is why I chose the base cluster size to be 18.   
![image](https://github.com/user-attachments/assets/3a3596a2-bae2-4624-bad6-937653df0f06)
![image](https://github.com/user-attachments/assets/e6dfccf4-4585-40e5-baca-a1f531749f6b)

Affinity Propagation generates various cluster sizes for various PCA configurations.  
![image](https://github.com/user-attachments/assets/55362dfe-b212-41f2-b1c6-108f4a04faa0)


**Note** Here PCA starts at 3 and goes up to 768, therefore the lowest value of pca is 3, not 0.

### Number of Clusters
Now fixing number of Principal components at 3 and running the models for different cluster sizes generates the following loss metrics. Affinity propagation only generates one value along with the estimated number of clusters.
![image](https://github.com/user-attachments/assets/3e1c0fef-c2ef-4e2e-9bc8-6174fbc44bc9)
![image](https://github.com/user-attachments/assets/90690ba9-48af-4472-a53c-7c942b1c947e)

For Affinity Propagation:
```
Number of Selected features: 3 
Number of Clusters: 87 
Silhouette Score: 0.22066413903641394 
Davies Bouldin Score: 1.0432482481763137
```
And the Best metrics for all other models are as follows:
```
For model KMeans : Clusters: 3 - Silhouette Score: 0.30707533503582124 || Davies-Boulin Score: 1.168728660165156
For model Agglomerative Clustering : Clusters: 901 - Silhouette Score: 0.31233377139424795 || Davies-Boulin Score: 0.6823371172092239
For model Gaussian Mixture : Clusters: 3 - Silhouette Score: 0.3078791491633459 || Davies-Boulin Score: 1.1679644383455392
```

Further Hyper parameter tuning was done to find the best parameters for Agglomerative clustering, Gaussian Mixture and Affinity propagation. THe best model details are:
- **Agglomerative Clustering**:
  - **Ward** linkage with **901** clusters
  - *Silhouette Score*: `0.31233377139424795`
  - *Davies-Bouldin Score*: `0.6823371172092239`


## 4. Recommendations










