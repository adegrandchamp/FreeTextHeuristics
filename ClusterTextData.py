'''
Alexandra DeGrandchamp, Final Project
Text Clustering Module
'''

def create_tfidf(dataset, lower, tokens, stop_words=False):
    '''
        Creates a tf-idf matrix from a provided dataset (as list).
        Can toggle if text should be set to lower case
        Can toggle tokenization pattern
        Can toggle if an English-language stopwords list should be deployed (default is no)
        Returns the transformed sparse matrix and feature names for use in clustering
    '''
    from sklearn.feature_extraction.text import TfidfVectorizer

    if stop_words:
        stop = 'english'
    else:
        stop = None
    
    #for dataset used, lowercase, stop_words, and token_pattern are only relevant parameters
    tfidf = TfidfVectorizer(lowercase=lower,stop_words=stop,token_pattern=tokens)
    training_data = tfidf.fit_transform(dataset)

    features = feature_names(tfidf)

    return training_data, features

def feature_names(tfidf):
    '''
        Returns feature names of tfidf matrix
        For use to personalize clustering
    '''
    return tfidf.get_feature_names_out()

def create_svd(components,random,training_data):
    '''
        Create Latent Semantic Analysis using singular value decomposition from scikit-learn
        Takes as input the number of components, the desired random state, and a sparse tfidf matrix
        Returns transformed matrix with specified components, explained variance, and model
    '''

    from sklearn import decomposition

    svd = decomposition.TruncatedSVD(n_components=components, random_state=random)
    transformed_set = svd.fit_transform(training_data)
    explained_variance = svd.explained_variance_ratio_.sum()

    return transformed_set, explained_variance, svd

def create_kmeans(k, random, initializations, iterations, training_data):
    '''
        Create kMeans clustering algorithm
        Takes as input the desired number of clusters, the random state, number of initializations, and number of iterations,
        as well as a training data set.
        Returns predicted values and clusterer
    '''
    from sklearn.cluster import KMeans
    clusterer = KMeans(n_clusters=k, random_state=random, n_init=initializations,max_iter=iterations)
    predicted_values = clusterer.fit_predict(training_data)

    return predicted_values, clusterer

def k_range(components,items):
    '''
        Helper function for model_tuning algorithm
        Takes as input the number of desired components and the number of divisions sought
        Returns start, stop, and step for range that evenly divides components
    '''

    min_k = round(components*0.1)
    max_k = components
    range_k = max_k-min_k
    step_k = round(range_k/items)

    return min_k, max_k, step_k

def model_tuning(components,divisions,data,sample_parameter,metric_parameter='cosine'):
    '''
        Tuning algorithm for finding silhouette scores of multiple k cluster values
        Creates kMeans classifier for the divisions specified, clusters data, and finds silhouette score
        Takes as input the number of components, the number of desired divisions, training data matrix,
        the sample parameter desired (to limit iterations), and the metric_parameter (default is cosine distance)
        Returns the silhouette values
        Warning: the silhouette score runs in O(n^2) time; large data sets will take some time. Tuner will print milestones.
    '''
    from sklearn.metrics import silhouette_score

    silhouette_values = {}
    
    k_min, k_max, k_step = k_range(components,divisions)
    k_range_list = list(range(k_min,k_max+1,k_step))

    for k in k_range_list:
        preds,kmeans = create_kmeans(k,27,2,2,data) #initializations and iterations are hard-coded in to limit run-time
        sil_score = silhouette_score(data,preds,sample_size=sample_parameter,metric=metric_parameter)
        silhouette_values[k] = sil_score
        print(k,' of ',k_max,' clustering complete')
    
    return silhouette_values 

def model_tuning_no_svd(k_range, data, sample_parameter, metric_parameter='cosine'):
    '''
        Tuning algorithm for finding silhouette scores of multiple cluster values
        Designed to serve as "control" when tuning; this is not meant to accept SVD models (just pure tf-idf)
        K-range for testing should be specified, as no helper function is used
    '''
    from sklearn.metrics import silhouette_score

    silhouette_values = {}
    k_max = k_range[-1]

    for k in k_range:
        preds,kmeans = create_kmeans(k,27,2,2,data) #initializations and iterations are hard-coded in to limit run-time
        sil_score = silhouette_score(data,preds,sample_size=sample_parameter,metric=metric_parameter)
        silhouette_values[k] = sil_score
        print(k,' of ',k_max,' clustering complete')
    
    return silhouette_values 

def samples(data,percent_integer):
    '''
        Takes as input an array or sparse matrix and the desired percent as integer to sample
        Percent_integer should be an integer value (e.g. 1 for 1%)
        Returns the sample of rows
        Intended use case: limiting run time of particularly inefficent evaluation metrics, such as silhouettes
    '''

    n_rows = data.shape[0]
    pct = percent_integer/100
    sample_size = round(n_rows*pct)

    return sample_size

def transform_weights(kmeans, svd):
    '''
        Helper function for create_cluster_names
        Returns absolute value of weights vis a vis kmeans centroids and svd components
    '''
    import numpy as np

    weights = np.dot(kmeans.cluster_centers_,svd.components_)
    weights_abs = np.abs(weights)

    return weights_abs

def create_cluster_names(kmeans, svd, topN, features):
    '''
        Takes as input kmeans and svd models, as well as desired top N and the features vector from tfidf
        Calculates weights, creates dictionary of topN values
        Transforms values into formatted string
        Returns dataframe for joining into working data set
    '''
    import numpy as np

    weights = transform_weights(kmeans,svd)
    topN_mod = topN * -1

    cluster_term_dict = {}
    for i in range(kmeans.n_clusters):
        top = np.argsort(weights[i])[topN_mod:]
        cluster_term_dict[i] = [features[rank] for rank in top]
    
    cluster_labels_dict = {}
    for key in cluster_term_dict.keys():
        #if I'm very honest I don't know how to do this dynamically and I'm out of time to figure it out
        #so this is hard-coded for now until I can revisit
        rank1,rank2,rank3 = cluster_term_dict[key]
        label = f'{rank1}_{rank2}_{rank3}'
        cluster_labels_dict[key] = label
    
    return cluster_labels_dict

def create_df_clusters(predicted_clusters,column_name):
    '''
        Creates dataframe of predicted cluster numbers 
        For joining with text data
    '''
    import pandas as pd

    return pd.DataFrame(predicted_clusters,columns=[column_name])

def create_df_labels(dict,column_name):
    '''
        Creates dataframe of cleansed cluster labels
        For joining with text data
    '''
    import pandas as pd

    return pd.DataFrame(dict.values(),columns=[column_name],index=dict.keys())

def join_data(df1,df2,on_parameter=None):
    '''
        Simple join of two dataframes
        Can specify join criteria if not obvious
        Returns joined dataframe
    '''

    return df1.join(df2,on=on_parameter)
    

    