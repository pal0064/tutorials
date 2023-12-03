import streamlit as st
import pandas as pd
@st.cache_data()
def load_data(): 
    data = pd.read_parquet('https://raw.githubusercontent.com/pal0064/tutorials/master/streamlit_tutorial/housing_data_with_lat_long.parquet',columns= ['zip','period_begin','period_end','county','median_sale_price', 'homes_sold', 'inventory','irs_estimated_population','longitude','latitude'])
    data = data.dropna()
    data['period_begin'] = pd.to_datetime(data['period_begin'])
    data['period_end'] = pd.to_datetime(data['period_end'])
    int_columns = ['median_sale_price', 'homes_sold', 'inventory', 'irs_estimated_population']
    data[int_columns] = data[int_columns].astype(float)
    data = data[data['median_sale_price']>=150000]
    return data 
data = load_data()
st.sidebar.header("Filters")
select_by = st.sidebar.selectbox("Select Location filter:",['County','Zip'])
if select_by == 'County':
    selected_counties = st.sidebar.multiselect("Select Counties:", data['county'].unique(),default=['Pima County'])
    filtered_data = data[data['county'].isin(selected_counties)]
elif select_by =='Zip':
    selected_zip = st.sidebar.multiselect("Select Zip Code:", data['zip'].unique(),default=[85719])
    filtered_data = data[data['zip'].isin(selected_zip)]
value = st.sidebar.slider("Max Price", min_value = data['median_sale_price'].min( ),max_value = data['median_sale_price'].max( ),) 
filtered_data  = filtered_data[ filtered_data['median_sale_price'] <= value]
st.write(filtered_data)
