import openml
import pandas as pd


dataset = openml.datasets.get_dataset(554)

x, y, c, g = dataset.get_data(dataset_format="dataframe", target= "class")

print(x)
print(y)
print(c)
print(g)