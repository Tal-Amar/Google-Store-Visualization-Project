import pandas as pd
import json


def clean_dataset(df: pd.DataFrame):
    """
    Cleaning dataset. Removing useless columns & duplicates.
    :return: Cleaned DataFrame
    """
    df.drop_duplicates(subset=['App Id'], inplace=True)

    # Drop unused columns
    df.drop(['Android version Text',
             'Developer',
             'Developer Address',
             'Developer Internal ID',
             'Developer Id',
             'Developer Website',
             'Developer Email', 'Last update', 'Privacy Policy', 'Minimum Android',
             'Version'], axis=1, inplace=True)

    # Convert the 'Released' column to datetime
    df.loc[:, 'Released'] = pd.to_datetime(df['Released'], errors='coerce')

    # Convert the 'Rating' column to numeric
    df['Rating'] = pd.to_numeric(df['Rating'], errors='coerce')

    # Convert the 'Released' column to datetime
    df['Year'] = pd.to_datetime(df['Released'], errors='coerce').dt.year

    # Convert the 'Price' column to numeric
    df['Price'] = pd.to_numeric(df['Price'], errors='coerce')

    # Convert the 'Rating' column to numeric
    df['Minimum Installs'] = pd.to_numeric(df['Minimum Installs'], errors='coerce')

    # Convert NA to False
    df['Ad Supported'] = df['Ad Supported'].fillna(False)

    # Using suffix for installs. exp: "1,000+" => "1K+"
    df['Installs'] = df['Installs'].str.replace(',000', 'K').str.replace('KKK', 'B').str.replace('KK', 'M').str.replace("+", "")

    # Define the mapping for 'Content Rating'
    rating_mapping = {
        'Mature 17+': '17+',
        'Everyone 10+': '10+',
        'Adults only 18+': '18+',
        'Unrated': 'Everyone'
    }
    df['Content Rating'] = df['Content Rating'].replace(rating_mapping)

    # Convert 'Price' by 'Currency' USD using currency's dict
    with open('data/conversion_currencies.json', 'r') as file:
        conversion_currencies = json.load(file)

    df['conversion_currency'] = df['Currency'].map(conversion_currencies)
    df['Price_USD'] = df['Price'] * df['conversion_currency']
    df.drop('conversion_currency', axis=1, inplace=True)

    return df


if __name__ == "__main__":
    data = pd.read_csv("data/Playstore_final.csv", error_bad_lines=False)
    data = data.iloc[:, :29]  # Empty columns' not in use
    df_cleaned = clean_dataset(data)
    df_cleaned.to_csv("data/Playstore_final_processed.csv")
