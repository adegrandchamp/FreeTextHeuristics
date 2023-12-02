"""
Alexandra DeGrandchamp, Final Project
Clustering Main file
"""
import CleanseTextData as ct
import ClusterTextData as clust

file_path = '/Users/alexandradegrandchamp/Documents/GradSchool/DSC478/FinalProject/final_dataset.csv'

text_data, full_data = ct.isolate_text(file_path,'Item Full')

processed_data, final_data = ct.process_words(text_data)

training, features = clust.create_tfidf(final_data, lower=False,tokens=r"(?u)\b\w\w\w+\b", stop_words=True)

#hard coding this because otherwise the model tuner may crash due to sheer amount of data to process
#recommend if running model tuner at all to do so line by line
#paper includes results of this tuning/proof that functions work
transformed100,explained100,svd100 = clust.create_svd(100,27,training)
transformed200,explained200,svd200 = clust.create_svd(200,27,training)
transformed300,explained300,svd300 = clust.create_svd(300,27,training)
transformed500,explained500,svd500 = clust.create_svd(500,27,training)
transformed1000,explained1000,svd1000 = clust.create_svd(1000,27,training)
transformed1500,explained1500,svd1500 = clust.create_svd(1500,27,training)

one_percent = clust.samples(training,1) #to limit runtime of silhouette scores
k_range_control = clust.k_range(2000,10) #to create control (no SVD) kMeans cluster silhouette values

sils_svd_100 = clust.model_tuning(100, transformed100 , one_percent)
sils_svd_200 = clust.model_tuning(200, transformed200, one_percent)
sils_svd_300 = clust.model_tuning(300, transformed300, one_percent)
sils_svd_500 = clust.model_tuning(500,transformed500,one_percent)
sils_svd_1000 = clust.model_tuning(1000,transformed1000,one_percent)
sils_svd_1500 = clust.model_tuning(1500,transformed1500,one_percent)
sils_no_svd = clust.model_tuning_no_svd(k_range_control,training,one_percent)

final_components = 500
final_k = 350

final_transformed,final_explanation,final_svd = clust.create_svd(final_components,27,training)
final_preds,final_kmeans = clust.create_kmeans(final_k,27,5,10,final_transformed)

labels_dict = clust.create_cluster_names(final_kmeans,final_svd,3,features)

preds_df = clust.create_df_clusters(final_preds,'Cluster Number')
labels_df = clust.create_df_labels(labels_dict,'Cluster Labels')

cluster_df = clust.join_data(processed_data,preds_df)
final_df = clust.join_data(cluster_df,labels_df,'Cluster Number')

path= '/Users/alexandradegrandchamp/Documents/GradSchool/DSC478/FinalProject/cleansed_text_dataset.csv'
final_df.to_csv(path,index=False)