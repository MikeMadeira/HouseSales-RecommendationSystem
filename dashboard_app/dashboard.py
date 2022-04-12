import geopandas
import streamlit as st
import pandas as pd
import numpy as np
import folium

from PIL import Image
from streamlit_folium import folium_static
from folium.plugins import MarkerCluster
from datetime import datetime
import time

import plotly.express as px

    
def main():
    
    status = 'initial_page'
    new_status = set_page_header(status)
    
    if new_status == 'macro_analysis':
        # Extract Data
        path = '../data/kc_house_data.csv'
        data = get_data(path)
        
        url = 'https://opendata.arcgis.com/datasets/83fc2e72903343aabff6de8cb445b81c_2.geojson'
        geofile = get_geofile(url)

        # Data Transformation
        data['price_m2'] = data['price'] / (data['sqft_lot'] * 0.093)

        # make a safe deep copy
        data_copy = data.copy(deep=True)

        # Data Visualization
        visualize_overview(data_copy)
        statistics_view(data_copy)
        density_plot(data_copy, geofile)
        price_variation(data_copy)
        comercial_dist(data_copy)
        physical_attr_dist(data_copy)
        
    if new_status == 'report_analysis':
        # Extract Data
        path = '../data/kc_house_data.csv'
        data = get_data(path)
        
        url = 'https://opendata.arcgis.com/datasets/83fc2e72903343aabff6de8cb445b81c_2.geojson'
        geofile = get_geofile(url)
            
        # Data Transformation
        data['price_m2'] = data['price'] / (data['sqft_lot'] * 0.093)

        # make a safe deep copy
        data_copy = data.copy(deep=True)

        # Data Visualization
        visualize_overview(data_copy)

    set_page_footer()
    
    
def set_page_header(status):
    
    # Visualization Setup
    st.set_page_config(page_title='RE Invest', page_icon='re_icon.jpeg',
                   layout='wide', initial_sidebar_state='expanded')
    
    #Set header
    c1, c2 = st.columns((1, 5))

    # image
    with c1:
        photo = Image.open('real_estate_project_2.jpg')
        st.image(photo)
    
    # title and subtitle
    with c2:
        st.write('')
        HR_format = '<div> <p style="font-family:sans-serif;' \
                    'color:#000055 ;' \
                    'font-size: 40px;' \
                    'font-weight: bold;' \
                    'font-style: normal;' \
                    'text-align: left;">' \
                    'Real Estate Investment Recommendation Dashboard</p> </div>'
        st.markdown(HR_format, unsafe_allow_html=True)
    
    st.subheader("Feel free to change between dashboards:")
    
    if status == 'initial_page':
        if st.button('Macro Dashboard'):
            status = 'macro_analysis'
        if st.button('Report Dashboard'):
            status = 'report_analysis'

    
    return status

def set_page_footer():
    st.markdown('---')
    st.subheader('Dashboard App Purpose:')
    
    st.markdown('The **Macro Dashboard** allows for a business manager to check a data overview and do some basic statistical analysis.')
    
    st.markdown('The **Report Dashboard** allows for a business manager to check his required report on business problems presented last meeting.')
    st.write('')
    
    # Additional Info Section
    st.markdown('---')
    st.subheader("Data Analysis Project Info:")
    
    st.markdown('The macro and report dashboards visualization is part of [Real Estate Investment Recommendation System](https://github.com/MikeMadeira/HouseSales-RecommendationSystem) project on github made by **Michael Madeira**.')
    st.markdown('The main results and procedure can be found on the link above.')
    st.write('')
    st.write('Real Estate Recommendation Dashboard is a recommendation system for Real Estate companies based on data analysis and insights extraction which was tailored to find actionable insights and therefore solutions to the real estate business experts specific problems.')
    st.write('')
    st.markdown('Other Projects: [DSPortfolio](https://github.com/MikeMadeira)')
    st.markdown('Contact me: [LinkedIn](https://www.linkedin.com/in/michael-madeira-7b4350a7/)')
    
@st.cache(allow_output_mutation=True)
def get_data(path):
    data = pd.read_csv(path)
    
    with st.spinner('Please wait...'):
        time.sleep(1)
            
    return data

@st.cache(allow_output_mutation=True)
def get_geofile(url):
    geofile = geopandas.read_file(url)
    return geofile


def visualize_overview(data):
            
    f_attributes = st.sidebar.multiselect('Enter columns', data.columns)
    f_zipcode = st.sidebar.multiselect('Enter zipcode', data['zipcode'].unique())

    st.title('Data Overview')

    if ((f_zipcode != []) & (f_attributes != [])):
        data_overview = data.loc[data['zipcode'].isin(f_zipcode),f_attributes]
    elif ((f_zipcode == []) & (f_attributes != [])):
        data_overview = data.loc[:,f_attributes]
    elif ((f_zipcode != []) & (f_attributes == [])):
        data_overview = data.loc[data['zipcode'].isin(f_zipcode),:]
    else:
        data_overview = data

    st.write(data_overview.head())

    return None

def statistics_view(data):
    c1,c2 = st.columns((1,1))

    # Total and Average metrics
    df1 = data[['id','zipcode']].groupby('zipcode').count().reset_index()
    df2 = data[['price','zipcode']].groupby('zipcode').mean().reset_index()
    df3 = data[['sqft_living','zipcode']].groupby('zipcode').mean().reset_index()
    df4 = data[['price_m2','zipcode']].groupby('zipcode').mean().reset_index()

    # merge
    m1 = pd.merge(df1,df2,on='zipcode',how='inner')
    m2 = pd.merge(m1,df3,on='zipcode',how='inner')
    m3 = pd.merge(m2,df4,on='zipcode',how='inner')

    c1.header('Total and Averages per zipcode')
    c1.dataframe(m3,height=600)

    # Statistic Descriptive
    num_attributes = data.select_dtypes(include=['int64','float64'])
    media = pd.DataFrame(num_attributes.apply(np.mean))
    median = pd.DataFrame(num_attributes.apply(np.median))
    std = pd.DataFrame(num_attributes.apply(np.std))

    max_ = pd.DataFrame(num_attributes.apply(np.max))
    min_ = pd.DataFrame(num_attributes.apply(np.min))

    df_stats = pd.concat([max_, min_, media, median, std], axis=1).reset_index()

    df_stats.columns = ['attributes', 'max', 'min','mean','median','std']

    c2.header('Statistics Descriptive')
    c2.dataframe(df_stats,height=600)

    return None

def density_plot(data,geofile):
    # =======================
    # Densidade de Portfolio
    # =======================

    st.title( 'Region Overview' )

    c1, c2 = st.columns( ( 1, 1 ) )
    c1.header( 'Portfolio Density' )

    df = data.sample( 10 )

    # Base Map - Folium
    density_map = folium.Map(location=[data['lat'].mean(),
                              data['long'].mean() ],
                              default_zoom_start=15)
    marker_cluster = MarkerCluster().add_to(density_map)
    for name, row in df.iterrows():
        folium.Marker([row['lat'], row['long']],
                      popup='Sold R${0} on: {1}. Features: {2}, {3} bedrooms, {4} bathrooms, year built: {5}'.format(
                          row['price'],
                          row['date'],
                          row['sqft_living'],
                          row['bedrooms'],
                          row['bathrooms'],
                          row['yr_built']
                      )).add_to(marker_cluster)

    with c1:
        folium_static(density_map)

    # Region Price Map
    c2.header('Price Density')

    df = data[['price', 'zipcode']].groupby('zipcode').mean().reset_index()
    df.columns = ['ZIP', 'PRICE']

    df = df.sample(10)

    geofile = geofile[geofile['ZIP'].isin(df['ZIP'].tolist())]

    region_price_map = folium.Map(location=[data['lat'].mean(),
                                            data['long'].mean()],
                                  default_zoom_start=15)

    region_price_map.choropleth(data=df,
                                geo_data=geofile,
                                columns=['ZIP', 'PRICE'],
                                key_on='feature.properties.ZIP',
                                fill_color='YlOrRd',
                                fill_opacity=0.7,
                                line_opacity=0.2,
                                legend_name='AVG PRICE')

    with c2:
        folium_static(region_price_map)

    return None


def price_variation(data):
    # filters
    min_year_built = int(data['yr_built'].min())
    max_year_built = int(data['yr_built'].max())

    st.sidebar.subheader('Select Max Year Built')
    f_year_built = st.sidebar.slider('Year Built', min_year_built,
                                    max_year_built,
                                    min_year_built)

    st.header('Average Price per Year built')

    #data selection
    df = data.loc[data['yr_built'] < f_year_built]
    df = df[['yr_built','price']].groupby('yr_built').mean().reset_index()

    #plot
    fig = px.line(df, x='yr_built', y='price')
    st.plotly_chart(fig, use_container_width=True)


    ### Price Variation per Day

    st.header('Average Price per Day')
    st.sidebar.subheader('Select Max Date')

    # transform date attribute data type
    data.date = pd.to_datetime(data.date).dt.strftime('%Y-%m-%d')

    # filters
    min_date = datetime.strptime(data['date'].min(), '%Y-%m-%d')
    max_date = datetime.strptime(data['date'].max(), '%Y-%m-%d')

    f_date = st.sidebar.slider('Date', min_date,
                                       max_date,
                                       min_date)

    data.date = pd.to_datetime(data.date)

    #data selection
    df = data.loc[data['date'] < f_date]
    df = df[['date','price']].groupby('date').mean().reset_index()

    #plot
    fig = px.line(df, x='date', y='price')
    st.plotly_chart(fig, use_container_width=True)

    return None


def comercial_dist(data):

    st.header('Price Distribution')
    st.sidebar.subheader('Select Max Price')

    #filter
    min_price = int(data['price'].min())
    max_price = int(data['price'].max())
    avg_price = int(data['price'].mean())

    f_price = st.sidebar.slider('Price', min_price, max_price, avg_price)
    df = data.loc[data['price'] < f_price]

    # data plot
    fig = px.histogram(df, x='price', nbins=50)
    st.plotly_chart(fig, use_container_width=True)

    return None


def physical_attr_dist(data):

    st.sidebar.title('Attributes Options')
    st.title('Houses Distribution')

    #filters
    f_bed = st.sidebar.selectbox('Max number of bedrooms',
                                 sorted(set(data['bedrooms'].unique())))
    f_bath = st.sidebar.selectbox('Max number of bathrooms',
                                 sorted(set(data['bathrooms'].unique())))
    f_floors = st.sidebar.selectbox('Max number of floors',
                                 sorted(set(data['floors'].unique())))
    f_water = st.sidebar.checkbox('With water view')

    c1, c2 = st.columns(2)

    # House per bedrooms
    c1.header('Houses per bedrooms')
    df = data.loc[data['bedrooms'] < f_bed]
    fig = px.histogram(df, x='bedrooms', nbins=19)
    c1.plotly_chart(fig, use_container_width=True)

    # House per bathrooms
    c2.header('Houses per bathrooms')
    df = data.loc[data['bathrooms'] < f_bath]
    fig = px.histogram(df, x='bathrooms', nbins=12)
    c2.plotly_chart(fig, use_container_width=True)

    c1, c2 = st.columns(2)

    #House per floors
    c1.header('Houses per floors')
    df = data.loc[data['floors'] < f_floors]
    fig = px.histogram(df, x='floors', nbins=8)
    c1.plotly_chart(fig, use_container_width=True)

    #House per water view
    c2.header('Houses per water view')

    if f_water:
        df = data.loc[data['waterfront'] == 1]
    else:
        df = data

    fig = px.histogram(df, x='waterfront', nbins=10)
    c2.plotly_chart(fig, use_container_width=True)

if __name__ == '__main__':
    main()