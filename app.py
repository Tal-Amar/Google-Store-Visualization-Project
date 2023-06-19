import os
import pandas
import pandas as pd
import streamlit as st
from PIL import Image
import base64
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import nltk
from nltk.corpus import stopwords
from collections import defaultdict
from sklearn.feature_extraction.text import CountVectorizer

# Execute once, at the beginning
# Download stopwords only once
nltk.download('stopwords')
# Get the English stopwords
stop_words = list(stopwords.words('english'))
# Create a CountVectorizer instance with English stop words
vectorizer = CountVectorizer(stop_words=stop_words)


def clean_dataset(df: pandas.DataFrame):
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
    df['Installs'] = df['Installs'].str.replace(',000', 'K').str.replace('KKK', 'B').str.replace('KK', 'M')

    return df


def create_title():
    """
    Create head title, and sidebar head imj using CSS style

    """
    google_play_icon = Image.open(r"./design/google-play.png")
    st.set_page_config(page_title='Play-Store Dataset', page_icon=google_play_icon)
    # Define the background color for the title
    title_background_color = '#55B2FF'  # Change this to the desired color

    google_play_icon_path = r"./design/google-play.png"
    style = f"""
        <style>
            .title-container {{
                background-color: {title_background_color};
                padding: 10px;
                border-radius: 5px;
                display: flex;
                align-items: center;
            }}
            .title-text {{
                margin-left: 10px;
                color: white
            }}
        </style>
    """

    # Display the icon and title together
    st.markdown(style, unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class="title-container">
            <img src="data:image/png;base64,{base64.b64encode(open(google_play_icon_path, 'rb').read()).decode()}" alt="Google Play Store Icon" width="50" height="50">
            <h1 class="title-text">Play-Store Dataset</h1>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Create Sidebar icon
    B_W_icon_path = r"./design/B_W-icon.png"
    st.sidebar.markdown(f"""
        <img src="data:image/png;base64,{base64.b64encode(open(B_W_icon_path, 'rb').read()).decode()}" alt="Black & White Google Play Store Icon" width="50" height="50">
        <hr>
        """, unsafe_allow_html=True
                        )


def cumulative_bar_plot(df: pandas.DataFrame):
    """
    Create a chart of: Number of apps released until each year for each category.
    """

    # Draw separate line
    st.markdown("<hr>", unsafe_allow_html=True)

    # Extract the required columns, 'Year' created in 'clean_dataset' function.
    df_chart = df[['Category', 'Rating', 'Year']].copy()

    # Sidebar:
    st.sidebar.subheader("Number of Applicarion per Category")
    # Choosing number of categories to present (filter them latter)
    category_options = ['All', 'Top 5', 'Top 10', 'Bottom 5']
    selected_options = st.sidebar.selectbox("Select number of Categories", category_options)
    # Filter the dataframe based on the selected year range
    year_range = st.sidebar.slider(
        "Select Year Range",
        min_value=int(df_chart['Year'].min()),
        max_value=int(df_chart['Year'].max()),
        value=(int(df_chart['Year'].min()), int(df_chart['Year'].max()))
    )
    filtered_df = df_chart[df_chart['Year'].between(*year_range)]
    # Draw separate line
    st.sidebar.markdown("<hr>", unsafe_allow_html=True)

    # Cart:
    # Calculate the average rating per category
    avg_ratings = filtered_df.groupby('Category')['Rating'].mean().reset_index()

    # Group the data by category and count the number of applications
    grouped_df = filtered_df.groupby('Category').size().reset_index(name='Number of Applications')

    # Merge with average ratings
    grouped_df = pd.merge(grouped_df, avg_ratings, how='left', on='Category')

    # Filter the dataframe based on the selected category display option
    if selected_options == 'Top 5':
        filtered_grouped_df = grouped_df.nlargest(5, 'Number of Applications')
    elif selected_options == 'Top 10':
        filtered_grouped_df = grouped_df.nlargest(10, 'Number of Applications')
    elif selected_options == 'Bottom 5':
        filtered_grouped_df = grouped_df.nsmallest(5, 'Number of Applications')
    else:
        filtered_grouped_df = grouped_df

    # Create the bar chart using Plotly
    fig = px.bar(
        filtered_grouped_df,
        x='Category',
        y='Number of Applications',
        color='Rating',
        labels={'Number of Applications': 'Number of Applications', 'Rating': 'Average Rating'},
        title='Number of Applications per Category'
    )
    fig.update_layout(xaxis=dict(categoryorder='total descending', gridcolor='#dce2f7'),
                      yaxis=dict(gridcolor='#dce2f7'))

    # Display
    st.write(fig)


def box_line_plot(df: pandas.DataFrame):
    """
    Create a chart of: Number of Installs v.s Price
    """

    # Draw separate line
    st.markdown("<hr>", unsafe_allow_html=True)

    # Extract the required columns, 'Year' created in 'clean_dataset' function.
    df_chart = df[['Installs', 'Price', 'Free', 'App Name']].copy()

    # Filter only paid apps
    paid_df = df_chart[df_chart['Free'] == False]

    # Remove rows with NaN values
    paid_df.dropna(subset=['Price'], inplace=True)

    # Define order by number of installs
    x_order = ['0+', '1+', '5+', '10+', '50+', '100+', '500+',
               '1K+', '5K+', '10K+', '50K+', '100K+', '500K+',
               '1M+', '5M+', '10M+',
               '1B+', '5B+', '10B+']

    avg_price = df_chart.groupby('Installs')['Price'].mean().reset_index()
    avg_price = avg_price.sort_values(by='Installs', key=lambda x: pd.Categorical(x, categories=x_order))

    # Create a plot type button in the sidebar
    st.sidebar.subheader('Average Price / Price Distribution across Installs for Paid Apps')

    # Default present line chart
    if not st.session_state.get('plot_type'):
        st.session_state['plot_type'] = 'line'

    toggle_button = st.sidebar.button('Switch Plot Type')
    if toggle_button:
        if st.session_state.get('plot_type') == 'line':
            st.session_state['plot_type'] = 'box'
        else:
            st.session_state['plot_type'] = 'line'
    # Separate line
    st.sidebar.markdown("<hr>", unsafe_allow_html=True)

    # Create line or box plot
    if st.session_state.get('plot_type') == 'line':
        fig_1 = go.Figure()

        fig_1.add_trace(go.Scatter(
            x=avg_price['Installs'],
            y=avg_price['Price'],
            mode='lines',
            marker=dict(color='#51d8e5'),
            name='Average Price'
        ))
        fig_1.update_yaxes(showline=True, linecolor='white', mirror=True, zerolinecolor='white')
        fig_1.update_layout(
            title='Average Price across Installs for Paid Apps',
            xaxis=dict(title='Installs', categoryorder='array', categoryarray=x_order, gridcolor='#dce2f7'),
            yaxis=dict(title='Price', gridcolor='#dce2f7')  # Set the grid color for y-axis
        )
        st.write(fig_1)

    else:
        fig_2 = go.Figure()

        fig_2 = px.box(
            paid_df,
            x='Installs',
            y='Price',
            log_y=True,
            labels={'Price': 'Price (Log scale)', 'Installs': 'Number of Installs', 'App Name': 'App Name'},
            category_orders={'Installs': x_order},
            title='Price Distribution across Installs for Paid Apps'
        )
        fig_2.update_layout(yaxis_color='white', xaxis_color='white',
                            xaxis=dict(categoryorder='total descending', gridcolor='#dce2f7'),
                            yaxis=dict(gridcolor='#dce2f7'))  # Set the grid color for y-axis

        st.write(fig_2)


def words_scatter_plot(df: pandas.DataFrame):
    """
    Create a Scatter Plot among the top apps.
    For each word in the description: Number of Installs v.s Rating
    where the size is the frequency.
    """

    # Draw separate line
    st.markdown("<hr>", unsafe_allow_html=True)

    df_chart = df[['Rating', 'Minimum Installs', 'Summary']].copy()
    df_chart.dropna(inplace=True)



    # Sidebar:
    st.sidebar.subheader("Words' success in Top-50 Apps")
    # Choosing option for the minimum apps for a word - to present
    min_apps = st.sidebar.number_input('Enter minimum number of apps a word should appear in:', value=3, step=1)
    top_apps_options = ['50', '100', '500', '1000']
    apps_number = st.sidebar.selectbox("Number of Top Apps", top_apps_options)
    # Separate line
    st.sidebar.markdown("<hr>", unsafe_allow_html=True)

    # filter only top XX - most success (Installs and Rating combination)
    filtered_df = df_chart[df_chart['Minimum Installs'] > 10000000]
    # "title_top" - format for title, later on...
    top_words_ratings, title_top = filtered_df.nlargest(50, 'Rating'), 50
    # if apps_number == '50':
    #     top_words_ratings = filtered_df.nlargest(50, 'Rating')
    #     title_top = 50
    if apps_number == '100':
        top_words_ratings = filtered_df.nlargest(100, 'Rating')
        title_top = 100
    if apps_number == '500':
        top_words_ratings = filtered_df.nlargest(500, 'Rating')
        title_top = 500
    if apps_number == '1000':
        top_words_ratings = filtered_df.nlargest(1000, 'Rating')
        title_top = 1000

    # Reset the index of df to match the index of word_counts_df
    top_words_ratings = top_words_ratings.reset_index(drop=True)

    # Create a DataFrame of words' count
    word_counts = vectorizer.fit_transform(top_words_ratings['Summary'])
    word_counts_df = pd.DataFrame(word_counts.toarray(), columns=vectorizer.get_feature_names_out())
    # Convert word counts into a binary form
    word_exists_df = word_counts_df.astype(bool).astype(int)

    # Create dictionaries to store the calculated values for the df that will be present
    apps_count = defaultdict(int)
    total_rating = defaultdict(int)
    total_installs = defaultdict(int)

    # For each word it will calculate (temporary) the total sum of Rating and Installs
    for word in word_exists_df.columns:
        apps_count[word] = word_exists_df[word].sum()
        if apps_count[word] >= min_apps:
            total_rating[word] = (word_exists_df[word] * top_words_ratings['Rating']).sum()
            total_installs[word] = (word_exists_df[word] * top_words_ratings['Minimum Installs']).sum()

    # Convert the dictionaries to DataFrames
    count_df = pd.DataFrame(list(apps_count.items()), columns=['Word', 'apps_number'])
    rating_df = pd.DataFrame(list(total_rating.items()), columns=['Word', 'total_rating'])
    installs_df = pd.DataFrame(list(total_installs.items()), columns=['Word', 'total_installs'])

    # Merge the dataframes
    words_df = pd.merge(count_df, rating_df, on='Word')
    words_df = pd.merge(words_df, installs_df, on='Word')

    # Calculate the mean rating and mean installs for each word
    words_df['mean_rating'] = words_df['total_rating'] / words_df['apps_number']
    words_df['mean_installs'] = words_df['total_installs'] / words_df['apps_number']

    # Drop the total rating and total installs columns
    words_df = words_df.drop(columns=['total_rating', 'total_installs'])

    # Create DataFrame for the word's chart
    fig = px.scatter(
        words_df,
        x='mean_installs',
        y='mean_rating',
        size='apps_number',
        hover_data=['Word', 'apps_number', 'mean_rating', 'mean_installs'],
        labels={'mean_rating': 'Average Rating', 'mean_installs': 'Average Installs'},
        title=f"Words' success in Top-{title_top} Apps"
    )
    fig.update_layout(xaxis=dict(categoryorder='total descending', gridcolor='#dce2f7', showgrid=True),
                      yaxis=dict(gridcolor='#dce2f7', showgrid=True))

    # Display
    st.write(fig)


# TO DO: Delete if not needed
def scatter_subplots(df: pandas.DataFrame):
    """

    :return:
    """
    # Draw separate line
    st.markdown("<hr>", unsafe_allow_html=True)

    df_chart = df[['Rating', 'Minimum Installs', 'Installs', 'Ad Supported', 'In app purchases', 'Category']].copy()
    df_chart.dropna(inplace=True)

    # Sidebar:
    st.sidebar.subheader('Price vs. Installs with Facets: "In-App Purchases" and "Ad Supported"')
    # Choosing multiselect categories to present in char
    cat_apps = list(set(df_chart['Category'].to_list()))
    cat_ms = st.sidebar.selectbox("Categories", cat_apps)

    rating_range = st.sidebar.slider(
        "Select Rating Range",
        min_value=int(df_chart['Rating'].min()),
        max_value=int(df_chart['Rating'].max()),
        value=(int(df_chart['Rating'].min()), int(df_chart['Rating'].max()))
    )

    installs_range = st.sidebar.slider(
        "Select Installs Range",
        min_value=int(df_chart['Minimum Installs'].min()),
        max_value=int(df_chart['Minimum Installs'].max()),
        value=(int(df_chart['Minimum Installs'].min()), int(df_chart['Minimum Installs'].max()))
    )

    filtered_df = df_chart[(df_chart['Rating'].between(*rating_range)) &
                           (df_chart['Minimum Installs'].between(*installs_range))]
    # Separate line
    # st.sidebar.markdown("<hr>", unsafe_allow_html=True)

    # Filtered by chosen categories
    filtered_df = filtered_df[filtered_df['Category'] == cat_ms]

    # Create all subplots DataFrames. 'n' means "False"
    purchases_ad = filtered_df[(filtered_df['In app purchases'] == True) & (filtered_df['Ad Supported'] == True)]
    n_purchases_ad = filtered_df[(filtered_df['In app purchases'] == False) & (filtered_df['Ad Supported'] == True)]
    purchases_n_ad = filtered_df[(filtered_df['In app purchases'] == True) & (filtered_df['Ad Supported'] == False)]
    n_purchases_n_ad = filtered_df[(filtered_df['In app purchases'] == False) & (filtered_df['Ad Supported'] == False)]

    # Initialize figure with subplots
    fig = make_subplots(
        rows=2, cols=2, subplot_titles=("In-App Purchases" and "Ad Supported",
                                        "No In-App Purchases" and "Ad Supported",
                                        "In-App Purchases" and "No Ad Supported",
                                        "No In-App Purchases" and "No Ad Supported")
    )

    # Define data subsets
    datasets = [purchases_ad, n_purchases_ad, purchases_n_ad, n_purchases_n_ad]
    positions = [(1, 1), (1, 2), (2, 1), (2, 2)]

    for data, pos in zip(datasets, positions):
        categories = data['Category'].unique()
        for category in categories:
            data_subset = data[data['Category'] == category]
            fig.add_trace(go.Scatter(x=data_subset['Minimum Installs'], y=data_subset['Rating'],
                                     mode='markers', marker=dict(color='white'),
                                     name=category),
                          row=pos[0], col=pos[1])

    # Update xaxis properties
    fig.update_xaxes(title_text="Installs", row=1, col=1)
    fig.update_xaxes(title_text="Installs", row=1, col=2)
    fig.update_xaxes(title_text="Installs", row=2, col=1)
    fig.update_xaxes(title_text="Installs", row=2, col=2)

    # Update yaxis properties
    fig.update_yaxes(title_text="Rating", row=1, col=1)
    fig.update_yaxes(title_text="Rating", row=1, col=2)
    fig.update_yaxes(title_text="Rating", row=2, col=1)
    fig.update_yaxes(title_text="Rating", row=2, col=2)

    # Update title and height
    fig.update_layout(title_text='Price vs. Installs with Facets: "In-App Purchases" and "Ad Supported"',
                      height=700, showlegend=False)

    # Display
    st.write(fig)


def box_subplots(df: pandas.DataFrame):
    """
    Create a box plot of Rating distribution for chosen Category with 4 facets:
    Free, Ad Supported, In-App Purchases, Countent Rating.
    """
    # Draw separate line
    st.markdown("<hr>", unsafe_allow_html=True)

    # create a data frame for the specific category
    df.dropna(inplace=True)

    # Sidebar:
    st.sidebar.subheader('Box Plot of Parameters for Chosen Category')
    # Choosing multiselect categories to present in char
    cat_apps = list(set(df['Category'].to_list()))
    chosen_category = st.sidebar.selectbox("Categories", cat_apps)

    # Assuming df is your DataFrame and 'chosen_category' is your selected category
    df_category = df[df['Category'] == chosen_category]

    # Create a subplot
    fig = make_subplots(rows=2, cols=2, subplot_titles=("Free", "Ad Supported", "In-App Purchases", "Content Rating"))

    # Add trace for 'Free'
    fig.add_trace(go.Box(y=df_category[df_category['Free'] == True]['Rating'], name="Free: True"), row=1, col=1)
    fig.add_trace(go.Box(y=df_category[df_category['Free'] == False]['Rating'], name="Free: False"), row=1, col=1)

    # Add trace for 'Ad Supported'
    fig.add_trace(go.Box(y=df_category[df_category['Ad Supported'] == True]['Rating'], name="Ad Supported: True"),
                  row=1, col=2)
    fig.add_trace(go.Box(y=df_category[df_category['Ad Supported'] == False]['Rating'], name="Ad Supported: False"),
                  row=1, col=2)

    # Add trace of In app purchases
    fig.add_trace(
        go.Box(y=df_category[df_category['In app purchases'] == True]['Rating'], name="In-App Purchases: True"),
        row=2, col=1)
    fig.add_trace(
        go.Box(y=df_category[df_category['In app purchases'] == False]['Rating'], name="In-App Purchases: False"),
        row=2, col=1)

    # Add trace for 'Content Rating'
    for rating in df_category['Content Rating'].unique():
        fig.add_trace(
            go.Box(y=df_category[df_category['Content Rating'] == rating]['Rating'], name=f"Content Rating: {rating}"),
            row=2, col=2)

    # Update grid line color for all subplots, size and title
    fig.update_xaxes(gridcolor='#daeeff')
    fig.update_yaxes(gridcolor='#daeeff')
    fig.update_layout(height=600, width=800, title_text="Box Plot of Parameters for Chosen Category")



    # Display the plot
    st.write(fig)


if __name__ == "__main__":
    data = pd.read_csv("Playstore_final.csv", error_bad_lines=False)
    data = data.iloc[:, :29]
    df_cleaned = clean_dataset(data)
    create_title()
    # Chart 1
    cumulative_bar_plot(df_cleaned)
    # Chart 2
    box_line_plot(df_cleaned)
    # Chart 3
    words_scatter_plot(df_cleaned)
    # Chart 4
    # scatter_subplots(df_cleaned)
    box_subplots(df_cleaned)
