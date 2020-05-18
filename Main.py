import mathhelp
import numpy as np
import csv

with open('dataset_191_wine.csv', mode='r') as csv_file:
    csv_reader: csv.DictReader = csv.reader(csv_file, quoting=csv.QUOTE_NONNUMERIC) # from string to float
    column_counter = 0
    for row in csv_reader:
        list_of_rows = list(csv_reader)  # 1st col with col_names not part of

data_set = list_of_rows

for j in data_set: # remove results / 1st column
    del j[0]

for i in range(0, len(data_set)): # normalizing vectors
    data_set[i] = mathhelp.normalize(data_set[i])


