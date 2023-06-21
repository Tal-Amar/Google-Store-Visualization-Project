import pandas
import pandas as pd
import streamlit as st
from PIL import Image
import base64
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from collections import defaultdict
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import streamlit.components.v1 as components


def create_title():
    """
    Create head title, and sidebar head imj using CSS style
    """

    google_play_icon = Image.open(r"./design/google-play.png")
    st.set_page_config(page_title='Play-Store Dataset', page_icon=google_play_icon)
    # Define the background color for the title
    title_background_color = '#94B5F3'  # Change this to the desired color

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
        <hr>
        <div style='font-size: 16px;color: black;'> 
             Explore insightful analyses and visualizations of the app ecosystem to gain valuable insights that can help you make informed decisions for creating better applications. Uncover trends, correlations, and influences within categories, pricing, user ratings, word choices, and developer decisions, providing actionable knowledge for optimizing your own app development strategies in the Google Apps Store.
             </div>
        """,
        unsafe_allow_html=True
    )


def ChangeWidgetFontSize(wgt_txt, wch_font_size='12px'):
    """
    Change widget font size
    :param wgt_txt: The widget's text
    :param wch_font_size: Wanted text size
    """

    html_str = """<script>var elements = window.parent.document.querySelectorAll('p'), i;
                for (i = 0; i < elements.length; ++i) 
                    { if (elements[i].textContent.includes(|wgt_txt|)) 
                        { elements[i].style.fontSize ='""" + wch_font_size + """'; } }</script>  """

    html_str = html_str.replace('|wgt_txt|', "'" + wgt_txt + "'")
    components.html(f"{html_str}", height=0, width=0)


def cumulative_bar_plot(df: pd.DataFrame):
    """
    Create a chart of: Number of apps released until each year for each category.
    """

    # Draw separate line
    st.markdown("<hr>", unsafe_allow_html=True)

    # Main panel:
    st.subheader("Application Publications per Category (Top 10)")
    # Add chart details
    st.markdown("""
        <div style='font-size: 16px;color: black;'> 
            Explore the number of apps released per category, sorted in descending order, with color-coded average rankings. Adjust the year range for insightful trend analysis.
        </div>
        <br>
        """, unsafe_allow_html=True)
    # Choosing number of categories to present (filter them later)
    # Filter the dataframe based on the selected year range
    year_range = st.slider(
        "Select Year Range",
        min_value=int(df['Year'].min()),
        max_value=int(df['Year'].max()),
        value=(int(df['Year'].min()), int(df['Year'].max()))
    )
    ChangeWidgetFontSize("Select Year Range", '18px')

    filtered_df = df[df['Year'].between(*year_range)]
    # Draw separate line

    # Chart:
    # Calculate the average rating per category
    avg_ratings = filtered_df.groupby('Category')['Rating'].mean().reset_index()

    # Group the data by category and count the number of applications
    grouped_df = filtered_df.groupby('Category').size().reset_index(name='Number of Applications')

    # Merge with average ratings
    grouped_df = pd.merge(grouped_df, avg_ratings, how='left', on='Category')

    # Filter the dataframe based on the selected category display option
    filtered_grouped_df = grouped_df.nsmallest(10, 'Number of Applications')


    # Create the bar chart using Plotly
    fig = px.bar(
        filtered_grouped_df,
        x='Category',
        y='Number of Applications',
        color='Rating',
        labels={'Number of Applications': 'Number of Applications', 'Rating': 'Average Rating'}
    )

    fig.update_layout(
        xaxis=dict(
            categoryorder='total descending',
            gridcolor='#dce2f7',
            tickfont=dict(size=16, color='black'),
            title=dict(text='Category', font=dict(size=18, color='black'))  # Set font size for x-axis label title
        ),
        yaxis=dict(
            gridcolor='#dce2f7',
            tickfont=dict(size=16, color='black'),
            title=dict(text='Number of Applications', font=dict(size=18, color='black'))  # Set font size for y-axis label title
        )
    )
    # Add black frame to each bar
    fig.update_traces(marker=dict(line=dict(color='black', width=1)))

    fig.update_coloraxes(colorbar=dict(
        outlinecolor='black',
        outlinewidth=1,
        tickfont=dict(size=16, color='black'),
        title=dict(text='Average Rating', font=dict(size=16, color='black'))
        )
    )
    fig.update_layout(width=800, showlegend=False)

    # Display
    st.write(fig)


def lines_plot(df: pandas.DataFrame):
    """
    Create a chart of: Number of Installs v.s Price
    """

    st.markdown("<hr>", unsafe_allow_html=True)
    st.subheader("Average Rating and Number of Installs across Prices")
    st.markdown("""
        <div style='font-size: 16px;color: black;'> 
            Visualize the correlation between application price and number of installs, as well as price and average rating, using an interactive line plot. Hover to view simultaneous price and curve values, and toggle between the two lines for focused analysis.
             </div>
        """, unsafe_allow_html=True)

    # Extract the required columns, 'Year' created in 'clean_dataset' function.
    df['Minimum Installs'] = df['Minimum Installs'].apply(lambda x: np.log(x + 1))

    df_chart = df[df["Price"] <= 10]

    # Define the bin edges and labels
    bin_edges = [i * 0.1 - 0.01 for i in range(102)]

    # Apply bins to the DataFrame
    df_chart['Bins'] = pd.cut(df_chart['Price'], bins=bin_edges, labels=False)
    df_chart['Bin_Labels'] = df_chart['Bins'].apply(lambda x: x * 0.1)

    Rating_price = df_chart.groupby('Bin_Labels')['Rating'].mean().reset_index()
    Rating_price = Rating_price.sort_values(by='Bin_Labels')

    window_size = 6
    Rating_price['Smoothed_Rating'] = Rating_price['Rating'].rolling(window=window_size,
                                                                     min_periods=1).mean().reset_index(0, drop=True)
    Rating_price['Normalized_Smoothed_Rating'] = Rating_price['Smoothed_Rating'].apply(
        lambda x: (x - min(Rating_price['Smoothed_Rating'])) / (
                    max(Rating_price['Smoothed_Rating']) - min(Rating_price['Smoothed_Rating'])))
    Rating_price['Smoothed_Rating'] = Rating_price['Smoothed_Rating'].apply(lambda x: np.round(x, 3))

    Installs_price = df_chart.groupby('Bin_Labels')['Minimum Installs'].mean().reset_index()
    Installs_price = Installs_price.sort_values(by='Bin_Labels')

    # window_size=6
    Installs_price['Smoothed_Installs'] = Installs_price['Minimum Installs'].rolling(window=window_size,
                                                                                     min_periods=1).mean().reset_index(
        0, drop=True)
    Installs_price['Normalized_Smoothed_Installs'] = Installs_price['Smoothed_Installs'].apply(
        lambda x: (x - min(Installs_price['Smoothed_Installs'])) / (
                    max(Installs_price['Smoothed_Installs']) - min(Installs_price['Smoothed_Installs'])))
    Installs_price['Original_Minimum_Installs'] = Installs_price['Smoothed_Installs'].apply(
        lambda x: np.round(np.power(np.e, x) - 1, 3))

    # Create line plot
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=Rating_price['Bin_Labels'],
        y=Rating_price['Normalized_Smoothed_Rating'],
        mode='lines',
        marker=dict(color='#B3304C'),
        hovertemplate='%{text}',
        hoverlabel=dict(bgcolor='#EA738D'),
        text=Rating_price['Smoothed_Rating'],
        name='Average Rating'
    ))
    fig.add_trace(go.Scatter(
        x=Installs_price['Bin_Labels'],
        y=Installs_price['Normalized_Smoothed_Installs'],
        mode='lines',
        marker=dict(color='#224193'),
        hovertemplate='%{text}',
        hoverlabel=dict(bgcolor='#89ABE3'),
        text=Installs_price['Original_Minimum_Installs'],
        name='Average Installs'
    ))
    fig.update_yaxes(showline=True, linecolor='white', mirror=True, zerolinecolor='#dce2f7')
    fig.update_layout(hovermode="x")
    fig.update_layout(
        xaxis=dict(
            tickfont=dict(size=16, color='black'),
            title=dict(text='Price', font=dict(size=18, color='black')),
            categoryorder='array',
            gridcolor='#dce2f7'),
        yaxis=dict(
            tickfont=dict(size=16, color='black'),
            title=dict(text='Normalized succses', font=dict(size=18, color='black')),
            gridcolor='#dce2f7'),
        hoverlabel=dict(
            bgcolor="#cccccc",
            font_color="black",
            font_size=14
        )
    )
    fig.update_layout(legend=dict(font=dict(size=16)))
    fig.update_layout(width=800, showlegend=True)

    st.write(fig)


def words_scatter_plot(df: pandas.DataFrame):
    """
    Create a Scatter Plot among the top apps.
    For each word in the description: Number of Installs v.s Rating
    where the size is the frequency.
    """

    # Draw separate line
    st.markdown("<hr>", unsafe_allow_html=True)
    st.subheader("Word Analysis: Rating, Installs, and Frequency")
    st.markdown("""
        <div style='font-size: 16px;color: black;'> 
            Explore word correlations in top app descriptions through a scatter plot. Words are positioned based on average ratings and installs, with point size indicating appearance frequency. Adjust filters for minimum appearances and top app selection to gain insights into influential keywords.
            </div>
        <br>
        """, unsafe_allow_html=True)

    # Sidebar:
    # Choosing option for the minimum apps for a word - to present
    st.number_input('Minimum number of appearances:', value=3, step=1)
    top_apps_options = ['50', '100', '500', '1000']
    apps_number = st.selectbox("Number of Top Apps:", top_apps_options)
    ChangeWidgetFontSize("Minimum number of appearances:", '18px')
    ChangeWidgetFontSize("Number of Top Apps:", '18px')


    # filter only top XX - most success (Installs and Rating combination)
    filtered_df = df[df['Minimum Installs'] > 10000000]
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

    with open('data/stop_words.txt', 'r') as f:
        stop_words = [line.strip() for line in f]

    vectorizer = CountVectorizer(stop_words=stop_words)
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
        labels={'mean_rating': 'Average Rating', 'mean_installs': 'Average Installs'}
    )
    fig.update_layout(
        xaxis=dict(
            categoryorder='total descending',
            tickfont=dict(size=16, color='black'),
            title=dict(font=dict(size=18, color='black')),
            gridcolor='#dce2f7', showgrid=True
        ),
        yaxis=dict(
            gridcolor='#dce2f7',
            tickfont=dict(size=16, color='black'),
            title=dict(font=dict(size=18, color='black')),
            showgrid=True
        )
    )

    # Display
    st.write(fig)


def box_subplots(df: pandas.DataFrame):
    """
    Create a box plot of Rating distribution for chosen Category with 4 facets:
    Free, Ad Supported, In-App Purchases, Countent Rating.
    """

    # Draw separate line
    st.markdown("<hr>", unsafe_allow_html=True)
    st.subheader('Developer Decision Impact on Average Rating')
    st.markdown("""
        <div style='font-size: 16px;color: black;'> 
            Analyze the effect of app developer decisions on average ratings through an interactive box plot. Adjust the category selection to explore specific decision factors and their influence on user ratings.
            </div>
        <br>
        """, unsafe_allow_html=True)

    # Choosing multiselect categories to present in char
    cat_apps = list(set(df['Category'].to_list()))
    chosen_category = st.selectbox("Categories", cat_apps)

    # Assuming df is your DataFrame and 'chosen_category' is your selected category
    df_category = df[df['Category'] == chosen_category]

    # Create a subplot
    fig = make_subplots(rows=2, cols=2, subplot_titles=("Free", "Ad Supported", "In-App Purchases", "Content Rating"))

    parameters = ['Free', 'Ad Supported', 'In app purchases']
    places = [(1, 1), (1, 2), (2, 1)]

    for parameter, place in zip(parameters, places):
        fig.add_trace(go.Box(
            y=df_category[df_category[parameter] == True]['Rating'], name="True", line_color='#89ABE3'),
            row=place[0], col=place[1])
        fig.add_trace(go.Box(
            y=df_category[df_category[parameter] == False]['Rating'], name="False", line_color='#EA738D'),
            row=place[0], col=place[1])
        fig.update_xaxes(
            gridcolor='#000000',
            tickfont=dict(size=16, color='black'),
            title=dict(font=dict(size=18, color='black')),
            row=place[0], col=place[1]
        )
        fig.update_yaxes(
            gridcolor='#000000',
            tickfont=dict(size=16, color='black'),
            title=dict(font=dict(size=18, color='black')),
            row=place[0], col=place[1]
        )

    # Create a color palette for 'Content Rating'
    unique_ratings = df_category['Content Rating'].unique()
    colors = ['#440154', '#3b528b', '#21918c', '#5ec962', '#fde725']
    palette = dict(zip(unique_ratings, colors))

    # Add trace for 'Content Rating'
    for rating in df_category['Content Rating'].unique():
        fig.add_trace(
            go.Box(
                y=df_category[df_category['Content Rating'] == rating]['Rating'],
                name=f"{rating}",
                marker=dict(color=palette[rating])),
            row=2,
            col=2
        )

    # Update grid line color for all subplots, size and title
    fig.update_xaxes(gridcolor='#daeeff',
                     tickfont=dict(size=16, color='black'))
    fig.update_yaxes(gridcolor='#daeeff',
                     tickfont=dict(size=16, color='black'))
    fig.update_layout(height=600, width=800, showlegend=False)

    fig.update_annotations(font_size=20, font_color="black")
    # Display the plot
    st.write(fig)


if __name__ == "__main__":
    create_title()
    # Chart 1
    chart1_df = pd.read_csv("data/cumulative_bar_plot.csv")
    cumulative_bar_plot(chart1_df)
    # Chart 2
    chart2_df = pd.read_csv("data/lines_plot.csv")
    lines_plot(chart2_df)
    # Chart 3
    chart3_df = pd.read_csv("data/words_scatter_plot.csv")
    words_scatter_plot(chart3_df)
    # Chart 4
    chart4_df = pd.read_csv("data/box_subplots.csv")
    box_subplots(chart4_df)
