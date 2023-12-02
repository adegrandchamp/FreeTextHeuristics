#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Alexandra DeGrandchamp, Final Project
Cleaning Main file
"""

import CleanData as cd

file_path = '/Users/alexandradegrandchamp/Documents/GradSchool/DSC478/FinalProject/PURCHASE ORDER DATA EXTRACT 2012-2015_0.csv'
relevant_fields = ['Acquisition Type','Sub-Acquisition Type', 'Department Name', 
                   'Item Name', 'Item Description','Total Price', 'Commodity Title',
                   'Class Title','Family Title','Segment Title']

imported_data = cd.read_file(file_path, relevant_fields)

counting_fields = ['Department Name', 'Commodity Title', 'Segment Title']

categorical_counts = cd.return_categorical_counts(imported_data, counting_fields)

remove_nan_fields = ['Acquisition Type', 'Item Name', 'Item Description', 'Commodity Title']
cleanse_nan_fields = ['Sub-Acquisition Type']

data_no_nan = cd.drop_na_values(imported_data, remove_nan_fields, cleanse_nan_fields)

filter_department = cd.filter_on_item_counts(data_no_nan, 'Department Name', categorical_counts[0], 20)
filter_commodity = cd.filter_on_item_counts(filter_department, 'Commodity Title', categorical_counts[1], 20)
filter_segment = cd.filter_on_item_counts(filter_commodity, 'Segment Title', categorical_counts[2], 1000)

text_fields = ['Item Name', 'Item Description']

modified_data = cd.create_text_column(filter_segment,text_fields,'Item Full')

final_dataset = cd.cleanse_currencies(modified_data, 'Total Price','$')

path= '/Users/alexandradegrandchamp/Documents/GradSchool/DSC478/FinalProject/final_dataset.csv'
final_dataset.to_csv(path,index=False)
