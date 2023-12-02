"""
Alexandra DeGrandchamp, Final Project
Tree Main file
"""

import BuildDecisionTree as bt

rs = 27 #random state for builds
test_pct = 0.3 #percent to use for model testing
file_path_final = '/Users/alexandradegrandchamp/Documents/GradSchool/DSC478/FinalProject/final_dataset.csv'
file_path_text = '/Users/alexandradegrandchamp/Documents/GradSchool/DSC478/FinalProject/cleansed_text_dataset.csv'
complete_data = bt.read_files(file_path_final,file_path_text)

class_labels = ['Segment Title','Family Title','Class Title','Commodity Title']
class_lookup, data_unclassified = bt.create_lookup_and_remove(complete_data,class_labels)

training_full,testing_full,target_train,target_test = bt.split_train_test(data_unclassified,class_lookup,test_percent=test_pct,random=rs)

#creating indexed lookups for spend value and text info
spend_lookup_train, train_no_spend = bt.create_lookup_and_remove(training_full,['Total Price'])
spend_lookup_test, test_no_spend = bt.create_lookup_and_remove(testing_full,['Total Price'])

text_labels = ['Item Full_df1','Item Full','Cluster Number']
text_lookup_train, train_no_text = bt.create_lookup_and_remove(train_no_spend,text_labels)
text_lookup_test, test_no_text = bt.create_lookup_and_remove(test_no_spend,text_labels)

#creating dummy variables for training data
training_with_dummies = bt.transform_to_dummies(train_no_text)
testing_with_dummies = bt.transform_to_dummies(test_no_text)

#also want to create a data set without the cluster labels as a control
control_train = train_no_text.drop('Cluster Labels',axis=1)
control_test = test_no_text.drop('Cluster Labels',axis=1)

control_train_dummies = bt.transform_to_dummies(control_train)
control_test_dummies = bt.transform_to_dummies(control_test)

#creating base test
base_test = bt.test_tree_full_hierarchy(target_train,training_with_dummies,target_test,testing_with_dummies,spend_lookup_test)

base_lvl0 = bt.unpack_accuracy(base_test,0)
base_lvl1 = bt.unpack_accuracy(base_test,1)
base_lvl2 = bt.unpack_accuracy(base_test,2)
base_lvl3 = bt.unpack_accuracy(base_test,3)

no_text_tree = bt.test_tree_full_hierarchy(target_train,control_train_dummies, target_test, control_test_dummies, spend_lookup_test)
notext_lvl0 = bt.unpack_accuracy(no_text_tree,0)
notext_lvl1 = bt.unpack_accuracy(no_text_tree,1)
notext_lvl2 = bt.unpack_accuracy(no_text_tree,2)
notext_lvl3 = bt.unpack_accuracy(no_text_tree,3)

max_depth_test1 = bt.parameter_tuning_num(1,20,5)
min_samples_leaf_test1 = bt.parameter_tuning_num(1,30,5)
min_samples_split_test1 = bt.parameter_tuning_num(2,20,5)

max_depth_test2 = bt.parameter_tuning_num(20,100,10)
max_features_test2 = bt.parameter_tuning_num(0.05,1,10)

parameter_dict_1 = {
    'criterion': ['entropy','gini'],
    'max_depth': max_depth_test1,
    'min_samples_leaf': min_samples_leaf_test1,
    'min_samples_split': min_samples_split_test1
    }

parameter_dict_2 = { 
    'criterion': ['entropy','gini'],
    'max_depth': max_depth_test2,
    'max_features': max_features_test2
    }

parameter_dict_3 = {
    'criterion': ['entropy','gini'],
    'max_features': max_features_test2
    }

best_params_1,best_score_1 = bt.perform_grid_search(training_with_dummies,target_train['Commodity Title'],parameter_dict_1,5)
best_params_2,best_score_2 = bt.perform_grid_search(training_with_dummies,target_train['Commodity Title'],parameter_dict_2,5)
best_params_3,best_score_3 = bt.perform_grid_search(training_with_dummies,target_train['Commodity Title'],parameter_dict_3,5)

final_parameters = {
    'criterion': 'entropy',
    'max_features': 0.85
}

tuned_tree_test = bt.test_tree_full_hierarchy(target_train,training_with_dummies,target_test,testing_with_dummies,spend_lookup_test,final_parameters)

tuned_lvl0 = bt.unpack_accuracy(tuned_tree_test,0)
tuned_lvl1 = bt.unpack_accuracy(tuned_tree_test,1)
tuned_lvl2 = bt.unpack_accuracy(tuned_tree_test,2)
tuned_lvl3 = bt.unpack_accuracy(tuned_tree_test,3)

type(0.05)






