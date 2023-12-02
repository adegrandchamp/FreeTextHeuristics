"""
Alexandra DeGrandchamp, Final Project
Build Decision Tree file
"""
import pandas as pd

def read_files(filepath1,filepath2):
    '''
        Reads in filepaths for two dataframes
        Joins dataframes, appending _df1 to first file if duplicate column names exist
        Returns joined files as a DataFrame
    '''
    
    dataset1 = pd.read_csv(filepath1)
    dataset2 = pd.read_csv(filepath2)
    final_dataset = dataset1.join(dataset2,lsuffix='_df1')

    return final_dataset

def create_lookup_and_remove(dataset,lookup_labels):
    '''
        Used if lookup tables are needed after classifying data or for evaluation
        Function accepts a dataset and a list of columns to serve as a lookup
        Returns lookup table and datset without lookup columns
    '''

    lookup = dataset[lookup_labels]
    data_no_lookup_vals = dataset.drop(lookup_labels,axis=1)

    return lookup, data_no_lookup_vals

def split_train_test(dataset,labels,test_percent,random=42):
    '''
        Splits dataset into specified train/test split, with training size = n_rows*(1-test_percent) and testing size = n_rows*test_percent
        Returns training set, testing set, targets for training set, and targets for testing set
    '''
    from sklearn.model_selection import train_test_split

    training,testing,target_train,target_test = train_test_split(dataset,labels,test_size=test_percent,random_state=random)

    return training,testing,target_train,target_test

def transform_to_dummies(dataset,specific_columns=None):
    '''
        Returns categorical dataset with dummy values
        If only specific columns are needed, pass list to specific_columns
    '''
    if specific_columns is None:
        dummy_data = pd.get_dummies(dataset)
    else:
        dummy_data = pd.get_dummies(dataset,columns=specific_columns)
    
    return dummy_data

def test_tree_full_hierarchy(train_labels,train_set,test_labels,test_set,test_values=None,parameter_dict=None):
    from sklearn import tree

    '''
        Function to create and test accuracy for multi-level classifications

        Takes as input multi-level training and testing label dataframes, training and test data sets,
            and a data frame of additional testing value, if desired

        Iterates through each column of a multi-column label set, creating decision tree for each

        Helper function creates accuracy calculation as well as 
            percentage of total for both correct and incorrect predictions, if second test_value is desired
            (Example: % of spend accurately classified at the level)
            Note: only one additional test value is supported
            
        Returns results of tests in dictionary with keys as index of label in hierarchy
    '''

    training_columns = train_labels.columns
    testing_columns = test_labels.columns
    rotations = len(training_columns)
    test_index = test_set.index

    tree_dict = {}
    for label_level in range(0,rotations):
        print('Testing level ',label_level)
        if parameter_dict is None:
            tree_classifier = tree.DecisionTreeClassifier()
        else:
            tree_classifier = tree.DecisionTreeClassifier(**parameter_dict)
        tree_trained = tree_classifier.fit(train_set,train_labels[training_columns[label_level]])
        tree_pred = tree_classifier.predict(test_set)
        accuracy_df = create_preds_eval(tree_pred,test_index,testing_columns[label_level],test_labels[testing_columns[label_level]],test_values)
        tree_dict[label_level] = [tree_trained,tree_pred,accuracy_df]
    
    return tree_dict

def create_preds_eval(pred_array,new_index,col_name,labels,non_count_values):
    '''
        Helper function for test_tree_full_hierarchy
        Aggregates data frame of accuracy and a custom value (e.g. dollars spent)
        Returns the resulting data frame
    '''

    pred_column_title = 'Predicted ' + col_name
    preds_as_df = pd.DataFrame(pred_array,index=new_index,columns=[pred_column_title])
    values_col = non_count_values.columns[0]

    if non_count_values is None:
        full_pred_df = preds_as_df.join(labels)
    else:
        full_pred_df = preds_as_df.join(labels).join(non_count_values)
        value_sum = full_pred_df[values_col].sum()
    
    full_pred_df['Accuracy'] = full_pred_df[pred_column_title]==full_pred_df[col_name]
    row_count = full_pred_df.count().iloc[0]
    full_pred_df['Pct Accuracy'] = full_pred_df.groupby('Accuracy')[col_name].transform(lambda x: 1/row_count)

    if non_count_values is None:
        accuracy_df = full_pred_df.groupby('Accuracy').agg({'Pct Accuracy':'sum'})
    else:
        full_pred_df['Pct Value'] = full_pred_df.groupby('Accuracy')[values_col].transform(lambda x: x/value_sum)
        accuracy_df = full_pred_df.groupby('Accuracy').agg({'Pct Accuracy':'sum','Pct Value': 'sum'})
    
    return accuracy_df

def unpack_accuracy(tree_dict,target_level):
    '''
        Helper function for test_tree_full_hierarchy
        Unpacks accuracy dataframes for inspection
    '''
    accuracy_df = tree_dict[target_level][2]
    
    return accuracy_df

def unpack_tree(tree_dict,target_level):
    '''
        Helper function for test_tree_full_hierarchy
        Unpacks tree models for inspection/further use
    '''
    tree_model = tree_dict[target_level][0]
    
    return tree_model

def feature_weights(tree_model):
    '''Returns feature weights for decision tree model'''
    return tree_model.feature_importances_

def perform_grid_search(training_data,training_labels,parameters,cross_validations=5):
    '''
        Performs a grid search for a training data set/labels and a specified parameter dictionary. 
        Also takes as input the number of cross-validations desired
        Returns the best parameters and score
    '''
    from sklearn import tree
    from sklearn.model_selection import GridSearchCV
    
    dummy_tree = tree.DecisionTreeClassifier()
    grid = GridSearchCV(dummy_tree,parameters,verbose=1,cv=cross_validations)
    grid.fit(training_data, training_labels)
    
    return grid.best_params_, grid.best_score_

def parameter_tuning_num(start,stop,step):
    '''
        Helper function for parameter tuning
        Used for numeric parameters only
        Accepts parameter string text, start, stop, and step of numeric value
        Returns a numpy array evenly spaced with arguments 
    '''
    import numpy as np

    if isinstance(start,float):
        data_type = 'float'
    elif isinstance(stop,float):
        data_type = 'float'
    else:
        data_type = 'int'

    return np.linspace(start,stop,step,dtype=data_type)