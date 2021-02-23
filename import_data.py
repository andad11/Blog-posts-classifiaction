import pandas as pd

#%%
df_blog = pd.read_csv('blogtext.csv')

#%%
def create_subset(df, n_rows, column):
    unique_values = df_blog[column].unique()

    df_list = []
    for value in unique_values:
        df_subset = df[df[column]==value].sample(n = int(n_rows/len(unique_values)), random_state = 0)
        df_list.append(df_subset)

    df_subset = pd.concat(df_list, axis = 0)

    return df_subset


#%% Choose subset
df_blog = create_subset(df_blog, n_rows= 100000, column='gender')




