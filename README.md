# Synopsis-Based-Manhwa-Recommender
A simple [Manhwa](https://en.wikipedia.org/wiki/Manhwa) Recommender based on a [Kaggle](https://www.kaggle.com/datasets/iridazzle/webtoon-originals-datasets?select=webtoon_originals_en.csv) dataset.

The recommnder is based on dataset of manhwa comics from kaggle. It uses [BERT](https://huggingface.co/docs/transformers/en/model_doc/bert) model to generate embeddings and uses averaged representations. The representation(768 dimensional vectors) are then used for feature selection using [PCA](https://www.ibm.com/think/topics/principal-component-analysis).
