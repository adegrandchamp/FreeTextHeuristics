'''
Alexandra DeGrandchamp, Final Project
Cleaning Dataset Module
'''
import pandas as pd
def read_file(filepath, relevant_fields):
    '''Reads in filepath of dataset and returns relevant columns in Pandas DataFrame'''
    
    dataset = pd.read_csv(filepath)
    filtered_dataset = dataset[relevant_fields]

    return filtered_dataset

def return_categorical_counts(dataset, relevant_fields):
    '''Reads in a Pandas Dataframe and fields required for value counts. Returns zipped object of all counts'''

    storage = {}
    index = 0
    for field in relevant_fields:
        storage[index] = dataset[field].value_counts()
        index += 1
    
    return storage

def drop_na_values(dataset, nan_remove_fields, na_fields):
    '''
        Takes as input a Pandas DataFrame, list of fields to remove NaN values, and list of fields to cleanse NaN to 'N/A'
        Cleanses NaN values from relevant fields in dataset
        Replaces fields that are fine with blank values with string 'N/A 
    '''
    dataset_no_nan = dataset.dropna(axis=0,subset=nan_remove_fields, how = 'any')
    na_fill = {}
    for field in na_fields:
        na_fill[field] = 'N/A'
    dataset_no_nan = dataset_no_nan.fillna(value=na_fill)
    
    return dataset_no_nan

def filter_on_item_counts(dataset, target_field, item_counter, min_value):
    '''
        Takes as input a Pandas DataFrame, the name of the target field to filter,
        an item-counting series (from return_categorical_counts), and the minimum number of occurrences desired.
        Returns DataFrame with only rows of field labels meeting the minimum
    '''
    filter = pd.Series(item_counter.loc[lambda x: x > min_value].keys())
    filtered_dataset = dataset[dataset[target_field].isin(filter)]

    return filtered_dataset

def cleanse_currencies(dataset, target_field, currency_symbol):
    '''
        Takes as input a Pandas DataFrame, the target field to cleanse, and a string of the currency symbol
        Strips field of currency symbol and converts field to float
        Returns cleansed DataFrame 
    '''
    cleansed_dataset = dataset.copy(deep=True)
    cleansed_dataset[target_field] = cleansed_dataset[target_field].str.replace(currency_symbol,'')
    cleansed_dataset[target_field] = cleansed_dataset[target_field].astype('float64')

    return cleansed_dataset

def create_text_column(dataset, target_columns, final_name):
    ''' 
        Takes as input a dataset and a list of target free-text columns. Also requests the final column name string.
        Concatenates free-text columns and drops original fields
        Returns modified dataset
    '''
    dataset[final_name] = dataset[target_columns[0]].str.cat(dataset[target_columns[1:]].astype(str),sep=' ')
    modified_data = dataset.drop(target_columns, axis=1)

    return modified_data