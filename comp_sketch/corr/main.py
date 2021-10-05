import pandas as pd

SELECTED_FEATURES = ['Eyeglasses', 'Male', 'Mustache', 'Smiling', 'Wavy_Hair', 'Young']

if __name__=='__main__':
    df = pd.read_csv('./list_attr_celeba.csv').set_index('img_id')[SELECTED_FEATURES]
    print(df.corr())

    new_df = pd.read_csv('./list_attr_celeba.csv').set_index('img_id').add(1).divide(2)
    print(new_df.head())

    print(new_df.sum(axis=0))