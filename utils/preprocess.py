import pandas as pd

def preprocess_input(data_dict, training_columns):
    input_df = pd.DataFrame([data_dict])
    input_df = pd.get_dummies(input_df)
    input_df = input_df.reindex(columns=training_columns, fill_value=0)
    return input_df

