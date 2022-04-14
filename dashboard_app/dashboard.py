import geopandas
import streamlit as st
import pandas as pd
import numpy as np
import math
import folium

from PIL import Image
from numerize.numerize import numerize
from streamlit_folium import folium_static
from folium.plugins import MarkerCluster
from datetime import datetime
import time

import plotly.express as px
from matplotlib import pyplot as plt

    
def main():
    
    status = 'initial_page'
    
    set_page_header(status)
    
    option = data_size_choice()
    
    if option != '':
        status = dashboard_choice(status)
    
    path = '../data/kc_house_data.csv'
    url = 'https://opendata.arcgis.com/datasets/83fc2e72903343aabff6de8cb445b81c_2.geojson'
        
    if status == 'macro_analysis':
        macro_dashboard(path,url,option)
        
    if status == 'report_analysis':
        report_dashboard(path,url,option)
        
    set_page_footer()
    

# ======================================================
# =============== PAGE CONFIG FUNCIONS =================
# ======================================================

def set_page_header(status):
    
    # Visualization Setup
    st.set_page_config(page_title='RE Invest', page_icon='re_icon.jpeg',
                   layout='wide', initial_sidebar_state='expanded')
    
    #Set header
    c1, c2 = st.columns((1, 5))

    # image
    with c1:
        photo = Image.open('real_estate_project.jpg')
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
                    'Real Estate Investment Dashboard</p> </div>'
        st.markdown(HR_format, unsafe_allow_html=True)
    
    return None

def set_page_footer():
    st.markdown('---')
    st.subheader('Dashboard App Purpose:')
    
    st.markdown('The **Macro Dashboard** allows for a business manager to check a data overview and do some basic statistical analysis.')
    
    st.markdown('The **Recommendation Report Dashboard** allows for a business manager to check his required report on business problems presented last meeting.')
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
    
    
def set_report_sidebar(data):
    
    # filters
    invest_opt = st.sidebar.multiselect('Investment Option', data.status.unique(), default=['to buy'])
    house_id = st.sidebar.multiselect('House Id', np.sort(data.id.unique()))
    f_zipcode = st.sidebar.multiselect('Enter zipcode', np.sort(data['zipcode'].unique()))
    f_attributes = st.sidebar.multiselect('Enter columns', data.columns, default=['id','zipcode','price','selling_price','profit_est','best_season_selling_price','best_season_profit_est','status'])
    f_living_size = st.sidebar.multiselect('Living Size', data.living_size.unique())
    f_lot_size = st.sidebar.multiselect('Lot Size', data.lot_size.unique())
    min_price = int(data.price.min())
    max_price = int(data.price.max())
#     st.sidebar.subheader('Select Min Price')
#     f_price = st.sidebar.slider('Price', min_price,
#                                               max_price,
#                                               min_price)
#     min_profit = int(data.profit_est.min())
#     max_profit = int(data.profit_est.max())
#     st.sidebar.subheader('Select Min Profit')
#     f_profit = st.sidebar.slider('Profit', min_profit,
#                                            max_profit,
#                                            min_profit)
    
    
    filters_dict = {'status':invest_opt,
                    'id':house_id,
                    'zipcode':f_zipcode,
                    'attributes':f_attributes,
                    'living_size':f_living_size,
                    'lot_size':f_lot_size}#,'price':f_price,'profit_est':f_profit}
    return filters_dict

def set_macro_sidebar(data):
    
    # filters
    f_attributes = st.sidebar.multiselect('Enter columns', data.columns)
    house_id = st.sidebar.multiselect('House Id', np.sort(data.id.unique()))
    f_zipcode = st.sidebar.multiselect('Enter zipcode', np.sort(data['zipcode'].unique()))
    f_order_column = st.sidebar.selectbox('Order Data Overview by column:',data.columns)
    f_ascending = st.sidebar.checkbox('Ascending')
    
    min_year_built = int(data['yr_built'].min())
    max_year_built = int(data['yr_built'].max())

    st.sidebar.subheader('Select Max Year Built')
    f_year_built = st.sidebar.slider('Year Built', min_year_built,
                                    max_year_built,
                                    max_year_built)
    
    # transform date attribute data type
    data.date = pd.to_datetime(data.date).dt.strftime('%Y-%m-%d')
    
    min_date = datetime.strptime(data['date'].min(), '%Y-%m-%d')
    max_date = datetime.strptime(data['date'].max(), '%Y-%m-%d')
    
    st.sidebar.subheader('Select Max Date')
    f_date = st.sidebar.slider('Date', min_date,
                                       max_date,
                                       max_date)
    
    min_price = int(data['price'].min())
    max_price = int(data['price'].max())
    avg_price = int(data['price'].mean())
    
    st.sidebar.subheader('Select Max Price')
    f_price = st.sidebar.slider('Price', min_price, max_price, max_price)
    
    st.sidebar.title('Attributes Options')
    
    f_bed = st.sidebar.selectbox('Max number of bedrooms',
                                 sorted(set(data['bedrooms'].unique())),index=len(data['bedrooms'].unique())-2)
    f_bath = st.sidebar.selectbox('Max number of bathrooms',
                                 sorted(set(data['bathrooms'].unique())),index=len(data['bathrooms'].unique())-1)
    f_floors = st.sidebar.selectbox('Max number of floors',
                                 sorted(set(data['floors'].unique())),index=len(data['floors'].unique())-1)
    f_water = st.sidebar.checkbox('With water view')
    
    st.sidebar.markdown('*For more information about Real Estate Investment Dashboard, please go to '
            '[Additional Information](#additional-information) section by the end of this page.*')
    
    filters_dict = {'id':house_id,
                    'zipcode':f_zipcode,
                    'attributes':f_attributes,
                    'yr_built':f_year_built,
                    'date':f_date,
                    'price':f_price,
                    'bedrooms':f_bed,
                    'bathrooms':f_bath,
                    'floors':f_floors,
                    'waterfront':f_water}
    return filters_dict, f_order_column, f_ascending

def dashboard_choice(status):

    st.write('If it is taking too long for the page to load, '
                        'please select a smaller database size above')
    
    st.markdown("**Feel free to change between dashboards:**")
    
    f_dashboard = st.selectbox('Choose Dashboard',['','Macro Dashboard','Report Dashboard'])
    if status == 'initial_page':
        if f_dashboard == 'Macro Dashboard':
            status = 'macro_analysis'
        if f_dashboard == 'Report Dashboard':
            status = 'report_analysis'
    
    return status

# =================================================
# =============== DASHBOARD FUNCTIONS =============
# =================================================

def macro_dashboard(path,url,option):
    
    # Extract Data
    data = get_data(path,option)
    geofile = get_geofile(url)

    # make a safe deep copy
    data_analysis = data.copy(deep=True)
    
    filters, f_order_column, f_ascending = set_macro_sidebar(data_analysis)
    
    # Pre-Process
    data_analysis = pre_processing(data_analysis)

    # create house total size
    data_analysis['house_total_m2'] = data_analysis['m2_living'] + data_analysis['m2_lot']

    # create price/m²
    data_analysis['price_m2'] = data_analysis['price']/data_analysis['house_total_m2']
    
    table_data = filter_data(data_analysis,filters,'table','macro')
    # Data Visualization
    visualize_overview(table_data, f_order_column, f_ascending)
    statistics_view(data_analysis)

    price_variation(data_analysis,filters['yr_built'],filters['date'])
    comercial_dist(data_analysis,filters['price'])
    physical_attr_dist(data_analysis,filters['bedrooms'],filters['bathrooms'],filters['floors'],filters['waterfront'])

#     st.title( 'Region Overview' )
#     c1, c2 = st.columns( ( 1, 1 ) )
    
#     portfolio_density(data_analysis,geofile,c1,'price','PRICE')
    

def report_dashboard(path,url,option):

    # Extract Data
    data = get_data(path,option)
    geofile = get_geofile(url)

    # make a safe deep copy
    report_data = data.copy(deep=True)

    # Data Transformation
    report_data = pre_processing(report_data)

    # Feature Engineering
    report_data = feature_engineering(report_data)
    
    # Set Report side bar
    filters = set_report_sidebar(report_data)

    # Report Overview
    report_overview(report_data)

    # Filter Data to report table
    report_table_data = filter_data(report_data,filters,'table')
    
    st.markdown('---')
    # Data Visualization
    visualize_report_table(report_table_data)

    # Filter Data to report maps
    report_maps_data = filter_data(report_data,filters,'maps')
    st.markdown('---')
    st.title( 'Region Overview' )
    c1, c2 = st.columns([2,2])
    portfolio_density(report_maps_data,geofile,c1,'profit_est','PROFIT')
    top_houses_per_profit(report_data,c2)
#     density_map(report_maps_data,geofile,c1)
#     cloropleth_map(report_maps_data,geofile,c2,'profit_est','PROFIT')

    st.markdown('---')
    st.title( 'Proposals for Renewals' )
    renovation_histograms(report_data)
    
    
# =================================================
# =============== DATA AND FILES FUNCTIONS =========
# =================================================   

@st.cache(allow_output_mutation=True)
def get_data(path, option):
    data = pd.read_csv(path)
    
    # selection of data sample
    data_25 = data.sample(math.floor(data.shape[0]*0.25))
    data_50 = data.sample(math.floor(data.shape[0]*0.5))
    data_75 = data.sample(math.floor(data.shape[0]*0.75))
    data_100 = data.sample(math.floor(data.shape[0]*1))
    
    data_reduced = []
    # filtering report data = data_r
    if option == '':
        data_reduced = []
    elif option == '25% of data':
        data_reduced = data_25
    elif option == '50% of data':
        data_reduced = data_50
    elif option == '75% of data':
        data_reduced = data_75
    elif option == '100% of data':
        data_reduced = data_100
            
    with st.spinner('Please wait...'):
        time.sleep(1)
            
    return data_reduced

@st.cache(allow_output_mutation=True)
def get_geofile(url):
    geofile = geopandas.read_file(url)
    return geofile

def data_size_choice():

    with st.container():
        st.subheader('Choose database size')
        
        # list of database sizes
        option = st.selectbox('Select size', ('', '25% of data', '50% of data', '75% of data', '100% of data'), key='data_size')
        
        # filtering report data = data_r
        if option == '':
            st.error('Choose database size on the list above to load the report')
        elif option == '25% of data':
            st.write('You have chosen 25% of the records')
        elif option == '50% of data':
            st.write('You have chosen 50% of the records')
        elif option == '75% of data':
            st.write('You have chosen 75% of the records')
        elif option == '100% of data':
            st.write('You have chosen the full database')
            st.warning('Please, note that choosing this option may slow down the report loading.')
        else:
            st.info('You must choose the database size')

    return option

def aux_filters_selection_mask(data,filters,report_element):
    
    empty_filter_mask = [True]*data.shape[0]
    final_mask = empty_filter_mask
    for feature in filters:
        if (filters[feature] != []) & (feature != 'attributes') & (report_element == 'table'):
            final_mask = np.logical_and(final_mask, data[feature].isin(filters[feature]))
        elif (filters[feature] != []) & (feature != 'attributes') & (feature != 'id') & (report_element != 'table'):
            final_mask = np.logical_and(final_mask, data[feature].isin(filters[feature]))
        else:
            final_mask = np.logical_and(final_mask, empty_filter_mask)
            
    return final_mask

def filter_data(data,filters,report_element,dashboard=''):
    
    filters_list = {key: filters[key] for key in filters if isinstance(filters[key],list)}

    mask = aux_filters_selection_mask(data,filters_list,report_element)
    
    if dashboard == 'macro':
        data_filtered = data
        
        house_id = filters_list['id']
        f_attributes = filters_list['attributes']
        if ((house_id != []) & (f_attributes != [])) & (report_element == 'table'):
            data_filtered = data_filtered.loc[mask & data['id'].isin(house_id),f_attributes]
        elif ((house_id == []) & (f_attributes != [])) & (report_element == 'table'):
            data_filtered = data_filtered.loc[mask,f_attributes]
        else:
            data_filtered = data_filtered.loc[mask,:]
        
        if 'bedrooms' in data_filtered.columns:
            data_filtered = data_filtered.loc[data_filtered.bedrooms < filters['bedrooms']]
        if 'bathrooms' in data_filtered.columns:
            data_filtered = data_filtered.loc[data_filtered.bathrooms < filters['bathrooms']]
        if 'floors' in data_filtered.columns:
            data_filtered = data_filtered.loc[data_filtered.floors < filters['floors']]
        if 'waterfront' in data_filtered.columns:
            data_filtered = data_filtered.loc[data_filtered.waterfront == filters['waterfront']]
        if 'yr_built' in data_filtered.columns:
            data_filtered = data_filtered.loc[data_filtered.yr_built < filters['yr_built']]
        if 'price' in data_filtered.columns:
            data_filtered = data_filtered.loc[data_filtered.price < filters['price']]
        if 'date' in data_filtered.columns:
            data_filtered = data_filtered.loc[data_filtered.date < filters['date']]
        
    else:
        data_filtered = data
        house_id = filters_list['id']
        f_attributes = filters_list['attributes']
        if ((house_id != []) & (f_attributes != [])) & (report_element == 'table'):
            data_filtered = data.loc[mask & data['id'].isin(house_id),f_attributes]
        elif ((house_id == []) & (f_attributes != [])) & (report_element == 'table'):
            data_filtered = data.loc[mask,f_attributes]
        else:
            data_filtered = data.loc[mask,:]

    return data_filtered.reset_index(drop=True)


# =================================================
# =============== TABLES FUNCTIONS =================
# =================================================

def visualize_overview(data, f_order_column, f_ascending):

    st.title('Data Overview')
    
    ascending = False
    if f_ascending:
        ascending = True
    st.write(data.sort_values(by=f_order_column,ascending=ascending))

    return None

def report_overview(data):
    st.title('Recommendation House Report')
    
    exp_overall = st.expander("Click here to expand/close overall information section.", expanded=True)
    with exp_overall:
        invested = data.loc[data.status == 'to buy','price'].sum()
        returned = data.loc[data.status == 'to buy','selling_price'].sum()
        profit = data.loc[data.status == 'to buy','profit_est'].sum()
        gross_revenue = (profit / invested) * 100
    
        c1, c2 = st.columns((1,3))
        with c1:
            st.header('Profit Overview')
            #st.subheader('Maximum Expected Profit')
            st.metric(label='Maximum Expected Profit', value=numerize(profit), delta=numerize(gross_revenue) + "%")
            #st.subheader('Maximum Value Invested')
            st.metric(label='Maximum Value Invested', value=numerize(invested))
            #st.subheader('Maximum Value Returned')
            st.metric(label='Maximum Value Returned', value=numerize(returned))
        with c2:
            # mainly insights
            st.header('Business Questions:')
            st.write('**1. Which houses should be bought and for what price?**')
            st.write('**2. Once its bought when it''s the best time period to sell it and for what price?**')
            st.write('**3. To rise the housing selling price, the company should do renewal works. So what would be good renewal changes?**')
            st.write('')
            st.subheader('Total properties on selected dataset: {:,}'.format(data.shape[0]))
            st.subheader('Total properties suggested to be purchased: {:,}'.format(data.loc[data.status == 'to buy'].shape[0]))
    
    
def visualize_report_table(data):

    st.header('Recommendation Houses Table')
        
    st.write(data)
    
    st.write('')
    st.write('The best season to sell and the respective expected selling price and profit for each house, can be extracted by changing the house selection through the filters on the sidebar')

    return None

# =======================================================================
# =============== VISUALIZATION OF INFORMATION FUNCIONS =================
# =======================================================================

def statistics_view(data):
    c1,c2 = st.columns((1,1))

    # Total and Average metrics
    df1 = data[['id','zipcode']].groupby('zipcode').count().reset_index()
    df2 = data[['price','zipcode']].groupby('zipcode').mean().reset_index()
    df3 = data[['m2_living','zipcode']].groupby('zipcode').mean().reset_index()
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

def portfolio_density(data,geofile,st_col,agg_feature,agg_feature_name):
    
    st_col.subheader( 'Recommended Houses Density Map' )
    
    df = data#.sample( 10 )

    # Base Map - Folium
    density_map = folium.Map(location=[data['lat'].mean(),
                              data['long'].mean() ],
                              default_zoom_start=15)
    marker_cluster = MarkerCluster().add_to(density_map)
    for name, row in df.iterrows():
        folium.Marker([row['lat'], row['long']],
                      popup='Sold R${0} on: {1}. Features: {2} m2_living, {3} bedrooms, {4} bathrooms, year built: {5}'.format(
                          row['price'],
                          row['date'],
                          row['m2_living'],
                          row['bedrooms'],
                          row['bathrooms'],
                          row['yr_built']
                      )).add_to(marker_cluster)
        
        
    df = data[[agg_feature, 'zipcode']].groupby('zipcode').mean().reset_index()
    df.columns = ['ZIP', agg_feature_name]

    df = df#.sample(10)

    geofile = geofile[geofile['ZIP'].isin(df['ZIP'].tolist())]

    folium.features.Choropleth(data=df,
                               geo_data=geofile,
                               columns=['ZIP', agg_feature_name],
                               key_on='feature.properties.ZIP',
                               fill_color='YlOrRd',
                               fill_opacity=0.7,
                               line_opacity=0.2,
                               legend_name='AVG '+agg_feature_name).add_to(density_map)
    
    with st_col:
        folium_static(density_map)

def top_houses_per_profit(data,st_col):
    
    columns = ['id','zipcode','price','selling_price','profit_est','best_season_selling_price','best_season_profit_est']
    with st_col:
        
        st.subheader('Top 10 Houses')
        st.write('With selling price suggestion increased based on:')
        st.write('Invested value above region median price value, house condition above 3 and best season to sell')
        st.write(data.loc[data.status=='to buy',columns].sort_values(by='best_season_profit_est',ascending=False).reset_index(drop=True).head(10))
    
def density_map(data,geofile,st_col):
    
    st_col.header( 'Recommended Houses Density Map' )

    df = data#.sample( 10 )

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
                          row['m2_living'],
                          row['bedrooms'],
                          row['bathrooms'],
                          row['yr_built']
                      )).add_to(marker_cluster)

    with st_col:
        folium_static(density_map)
        
def cloropleth_map(data,geofile,st_col,agg_feature,agg_feature_name):
    
    # Region Price Map
    st_col.header('Profit per Region Map')
    
    df = data[[agg_feature, 'zipcode']].groupby('zipcode').mean().reset_index()
    df.columns = ['ZIP', agg_feature_name]

    df = df#.sample(10)

    geofile = geofile[geofile['ZIP'].isin(df['ZIP'].tolist())]

    region_price_map = folium.Map(location=[data['lat'].mean(),
                                            data['long'].mean()],
                                  default_zoom_start=15)

    region_price_map.choropleth(data=df,
                                geo_data=geofile,
                                columns=['ZIP', agg_feature_name],
                                key_on='feature.properties.ZIP',
                                fill_color='YlOrRd',
                                fill_opacity=0.7,
                                line_opacity=0.2,
                                legend_name='AVG '+agg_feature_name)

    with st_col:
        folium_static(region_price_map)
    
    
def renovation_histograms(house_df: pd.DataFrame):

    ''' 
        For a specific zipcode region, returns a set of histograms, one 
        on each column for each living_size and on each row for 
        lot_size.
        On each graph the x-axis will be the amenity and the y-axis 
        the number of amenities.

        Parameters
        ----------
        zipcode : int
        amenity : string or list

        Returns
        ----------
        plots a set of histograms
    ''' 
    
    unique_zipcodes = np.sort(house_df['zipcode'].unique())
    zipcode = st.sidebar.selectbox('Enter the zipcode to check renovation proposals:',unique_zipcodes)
    
    st.sidebar.markdown('*For more information about this report, please go to '
            '[Additional Information](#additional-information) section by the end of this page.*')
    
    if zipcode in unique_zipcodes:
        
        zipcode_houses_df = house_df.loc[house_df.zipcode == zipcode,:]
        columns = zipcode_houses_df['living_size'].unique()
        rows = zipcode_houses_df['lot_size'].unique()
        
        houses_status_df = zipcode_houses_df.loc[(zipcode_houses_df.status == 'to buy') |
                                                 (zipcode_houses_df.status == 'to compare'),:]
        
        
        houses_to_buy_df = zipcode_houses_df.loc[zipcode_houses_df.status == 'to buy',:]
        houses_to_compare_df = zipcode_houses_df.loc[zipcode_houses_df.status == 'to compare',:]
        
        houses_to_buy_grouped = houses_to_buy_df.groupby(by=['living_size','lot_size']).median().reset_index()
        houses_to_compare_grouped = houses_to_compare_df.groupby(by=['living_size','lot_size']).median().reset_index()
        
        houses_to_buy_grouped.rename(columns = {'bedrooms':'houses_to_buy_bedrooms'}, inplace = True)
        houses_to_buy_grouped.rename(columns = {'bathrooms':'houses_to_buy_bathrooms'}, inplace = True)
        houses_to_compare_grouped.rename(columns = {'bedrooms':'houses_to_compare_bedrooms'}, inplace = True)
        houses_to_compare_grouped.rename(columns = {'bathrooms':'houses_to_compare_bathrooms'}, inplace = True)
        
        fig, ax = plt.subplots()
        houses_to_compare_grouped['property_size'] = houses_to_compare_grouped['living_size']+'_'+houses_to_compare_grouped['lot_size']
        houses_to_compare_grouped.plot.bar(x='property_size',y=['houses_to_compare_bedrooms','houses_to_compare_bathrooms'],ax=ax)
        houses_to_buy_grouped['property_size'] = houses_to_buy_grouped['living_size']+'_'+houses_to_buy_grouped['lot_size']
        houses_to_buy_grouped.plot.bar(x='property_size',y=['houses_to_buy_bedrooms','houses_to_buy_bathrooms'],ax=ax,color=['lightblue','orange'])
        
        c1, c2, c3 = st.columns([2,1,2])
        
        with c1:
            st.pyplot(fig)
        
        with c2:
            st.subheader('Result of labelling houses with investment option:')
            st.write(house_df.status.value_counts().rename('number of houses'))
            st.subheader('Conclusion')
            st.write('For on each zipcode, if the houses to compare, that were the ones above the median price and their condition were higher than 3, have more bathroom amenities or more bedrooms for houses with different living and lot sizes, comparing with the houses to buy, that need to have some renewal strategy to get an increase on value appreciation.')
            st.write('(Assuming that the only available amenities or houses features that are able to enhance the property are bathrooms and bedrooms.)')
        


def price_variation(data,f_year_built,f_date):

    st.header('Average Price per Year built')

    #data selection
    df = data.loc[data['yr_built'] < f_year_built]
    df = df[['yr_built','price']].groupby('yr_built').mean().reset_index()

    #plot
    fig = px.line(df, x='yr_built', y='price')
    st.plotly_chart(fig, use_container_width=True)


    ### Price Variation per Day

    st.header('Average Price per Day')

    data.date = pd.to_datetime(data.date)

    #data selection
    df = data.loc[data['date'] < f_date]
    df = df[['date','price']].groupby('date').mean().reset_index()

    #plot
    fig = px.line(df, x='date', y='price')
    st.plotly_chart(fig, use_container_width=True)

    return None


def comercial_dist(data,f_price):

    st.header('Price Distribution')

    df = data.loc[data['price'] < f_price]

    # data plot
    fig = px.histogram(df, x='price', nbins=50)
    st.plotly_chart(fig, use_container_width=True)

    return None


def physical_attr_dist(data,f_bed,f_bath,f_floors,f_water):

    st.title('Houses Distribution')

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

    
# ============================================================
# =============== DATA MANIPULATION FUNCIONS =================
# ============================================================

def pre_processing(df):
    
    # convert date type
    df.date = pd.to_datetime(df.date)
    
    # convert squared feet to squared meters of both living and lot size
    df[['m2_living','m2_above','m2_basement','m2_lot']] = df[['sqft_living','sqft_above','sqft_basement','sqft_lot']] * 0.0929

    df = df.drop(['sqft_living','sqft_above','sqft_basement','sqft_lot'], axis=1)
    
    return df

def feature_engineering(df):
    
    # create house total size
    df['house_total_m2'] = df['m2_living'] + df['m2_lot']
    
    # create price/m²
    df['price_m2'] = df['price']/df['house_total_m2']
    
    # discretize and categorize living_size, lot_size and house_type
    df['living_size'] = df['m2_living'].apply(lambda x: 'small_house' if x <= 135
                                                   else 'medium_house' if (x > 135) and (x <= 280)
                                                   else 'large_house')

    df['lot_size'] = df['m2_lot'].apply(lambda x: 'small_terrain' if x <= 500
                                             else 'medium_terrain' if (x > 500) and (x <= 1800)
                                             else 'large_terrain')
    df['house_type'] = df['house_total_m2'].apply(lambda x: 'apartment' if x <= 200
                                                       else 'villa' if (x > 200) and (x <= 400)
                                                       else 'townhouse' if (x > 400) and (x <= 1000)
                                                       else 'mansion' if (x > 1000) and (x <= 10000)
                                                       else 'countryhouse')
    
    # create the median price per region, living_size and lot_size
    zipcode_median_price = df[['price','zipcode','living_size','lot_size']].groupby(['zipcode','living_size','lot_size']).median().reset_index()
    zipcode_median_price.columns = ['zipcode','living_size','lot_size','region_median_price']

    df = pd.merge(df,zipcode_median_price,on=['zipcode','living_size','lot_size'],how='inner')
    
    # create the median price per m2 per region, living_size and lot_size
    zipcode_median_price_m2 = df[['price_m2', 'zipcode','living_size','lot_size']].groupby(['zipcode','living_size','lot_size']).median().reset_index()
    zipcode_median_price_m2.columns = ['zipcode','living_size','lot_size','region_median_price_m2']

    df = pd.merge(df,zipcode_median_price_m2,on=['zipcode','living_size','lot_size'],how='inner')
    
    # create the recommendation label feature as status
    df['status'] = df[['price','condition','region_median_price']].apply(lambda x: 'to buy' if (x[0] < x[2]) & (x[1] > 3)\
                                                                       else 'to consider' if (x[0] < x[2]) & (x[1] == 3)\
                                                                       else 'to compare' if (x[0] > x[2])  & (x[1] > 3)\
                                                                       else 'not worth buying',  axis = 1)
    
    # create percentage_value_below_median
    df['perc_value_below_median_price'] = (1 - (df['price']/df['region_median_price']))*100
    
    # set a selling price
    df['selling_price'] = df[['price','condition','status','perc_value_below_median_price']].apply(lambda x: x[0] if (x[1] == 3) & (x[2] == 'to consider')\
                                                                                                                     else x[0]*1.5 if (x[1] == 4) & (x[2] == 'to buy') & (x[3] > 50 and x[3] < 75)\
                                                                                                                     else x[0]*1.25 if (x[1] == 4) & (x[2] == 'to buy') & (x[3] > 25 and x[3] < 50)\
                                                                                                                     else x[0]*1.125 if (x[1] == 4) & (x[2] == 'to buy') & (x[3] > 0 and x[3] < 25)\
                                                                                                                     else x[0]*1.45 if (x[1] == 5) & (x[2] == 'to buy') & (x[3] > 50 and x[3] < 75)\
                                                                                                                     else x[0]*1.20 if (x[1] == 5) & (x[2] == 'to buy') & (x[3] > 25 and x[3] < 50)\
                                                                                                                     else x[0]*1.075 if (x[1] == 5) & (x[2] == 'to buy') & (x[3] > 0 and x[3] < 25)\
                                                                                                                     else x[0], axis=1)

    # Create profit variable
    df['profit_est'] = df['selling_price'] - df['price']
    
    # sort by percentage below median price, condition and profit estimation, descending, ascending and descending respectively
    sorted_df = df.sort_values(by=['perc_value_below_median_price', 'condition', 'profit_est'], ascending=[False,True,False])
    
    # Create a season column through date
    sorted_df['season'] = sorted_df[['price','date']].apply(lambda x: 'spring' if x[1].month in (4,5,6)
                                                                                                   else 'summer' if x[1].month in (7,8,9)
                                                                                                   else 'autumn' if x[1].month in (10,11,12)
                                                                                                   else 'winter', axis=1)

    sorted_df.loc[(sorted_df['date'].dt.month == 3) & (sorted_df['date'].dt.day > 19),'season'] = 'spring'
    sorted_df.loc[(sorted_df['date'].dt.month == 6) & (sorted_df['date'].dt.day > 20),'season'] = 'summer'
    sorted_df.loc[(sorted_df['date'].dt.month == 9) & (sorted_df['date'].dt.day > 21),'season'] = 'autumn'
    sorted_df.loc[(sorted_df['date'].dt.month == 12) & (sorted_df['date'].dt.day > 20),'season'] = 'winter'
    
    # Calculate the median price per season per region, living_size and lot_size
    groups_zip_sizes_season = sorted_df[['price', 'zipcode','living_size','lot_size','season']].groupby(['zipcode','living_size','lot_size','season'])
    median_price_per_season = groups_zip_sizes_season.median().reset_index()
    median_price_per_season.columns = ['zipcode','living_size','lot_size','season', 'median_price']
    unique_indices = median_price_per_season[['zipcode','living_size','lot_size']].drop_duplicates()
    best_season_df = pd.DataFrame(columns=['zipcode','living_size','lot_size','best_season','best_season_median_price'])
    for row in unique_indices.iterrows():

        rows_indeces = (median_price_per_season['zipcode'] == row[1]['zipcode']) & \
                       (median_price_per_season['living_size'] == row[1]['living_size']) & \
                       (median_price_per_season['lot_size'] == row[1]['lot_size'])

        unique_df = median_price_per_season.loc[rows_indeces]
        max_median_price = unique_df['median_price'].max()

        row_append = unique_df.loc[unique_df['median_price'] == max_median_price]
        row_append.columns = ['zipcode','living_size','lot_size','best_season','best_season_median_price']

        best_season_df = best_season_df.append(row_append,ignore_index=True)
    houses_best_selling_price = pd.merge(sorted_df,best_season_df,how='left',on=['zipcode','living_size','lot_size'])
    
    # set a best season selling price
    houses_best_selling_price['best_season_selling_price'] = houses_best_selling_price[['selling_price','best_season_median_price']].apply(lambda x: x[0]*1.05 if (x[0] < x[1])\
                                                                                                                                                else x[0], axis=1)
    houses_best_selling_price['best_season_profit_est'] = houses_best_selling_price['best_season_selling_price'] - houses_best_selling_price['price']
    return houses_best_selling_price


if __name__ == '__main__':
    main()