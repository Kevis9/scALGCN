import pandas as pd

mouse_df = pd.read_csv('input/query_raw_data.csv', index_col=0)
human_df = pd.read_csv('input/ref_raw_data.csv', index_col=0)

mouse_df.columns = human_df.columns
mouse_df.to_csv('input/query_raw_data.csv')

