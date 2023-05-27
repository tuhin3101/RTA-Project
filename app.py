import streamlit as st
import pandas as pd
import numpy as np
import joblib
# from sklearn.ensemble import ExtraTreesClassifier
from prediction import get_prediction, ordinal_encoder

model = joblib.load('randomForest.joblib')

st.set_page_config(page_title='Accident severity prediction app',layout='wide')

# creating option list for dropdown menu

option_light_condition = ['Daylight', 'Darkness - lights lit', 'Darkness - no lighting', 
                          'Darkness - lights unlit']

option_age_band = ['18-30', '31-50', 'Under 18', 'Over 51', 'Unknown']

option_day = ['Monday', 'Sunday', 'Friday', 'Wednesday', 'Saturday', 'Thursday', 'Tuesday']

option_road_surface = ['Dry', 'Wet or damp', 'Snow', 'Flood over 3cm. deep']

option_casualty_class = ['Driver or rider','Pedestrian', 'Passenger']

option_driving_experience = ['1-2yr', 'Above 10yr', '5-10yr', '2-5yr', 'No Licence', 'Below 1yr', 'unknown']

option_sex_of_casualty = ['Male', 'Female']

#option_no_of_casualities = [2, 1, 3, 4, 6, 5, 8, 7]

# option_no_of_vehicles = [2, 1, 3, 6, 4, 7]

# option_minute = [5, 10, 15, 30, 20, 40, 45, 35, 25, 0, 50, 55]

features = ['Light_condition', 'casualties', 'vehicle_involved', 'age_of_driver', 'day', 'road', 
 'casualty_class', 'driving_exp', 'sex_of_casualty'] 

st.markdown('Accident severity prediction application')

def main():
    with st.form('Prediction form'):

        st.subheader('Enter the inputs for following features')

        Light_condition = st.selectbox('Select Light_condition',options=option_light_condition)

        casualties = st.slider('Select no_of_casualities',1,8,step=1,format='%d')

        vehicle_involved = st.slider('Select no of vehicles involved',1,7,value=0,format='%d')
       
        age_of_driver = st.selectbox('Select age band of driver',options=option_age_band)

        minute = st.slider('Select Minute',0,55,step=5,format='%d')

        day = st.selectbox('Select day of week',options=option_day)

        road = st.selectbox('Select road surface condition',options=option_road_surface)

        casualty_class = st.selectbox('Select casualty class',options=option_casualty_class)

        driving_exp = st.selectbox('Select driving experience',options=option_driving_experience)

        sex_of_casualty = st.selectbox('Select sex of casualty', options=option_sex_of_casualty)


        submit = st.form_submit_button('Predict')


    if submit:

        Light_condition = ordinal_encoder(Light_condition, option_light_condition)
        
        age_of_driver = ordinal_encoder(age_of_driver, option_age_band)
        
        day = ordinal_encoder(day, option_day)
        
        road = ordinal_encoder(road, option_road_surface)
        
        casualty_class = ordinal_encoder(casualty_class, option_casualty_class)
        
        driving_exp = ordinal_encoder(driving_exp, option_driving_experience)
        
        sex_of_casualty = ordinal_encoder(sex_of_casualty, option_sex_of_casualty)

        
        data1 = np.array([Light_condition, casualties, vehicle_involved, age_of_driver, minute, day, road, 
                        casualty_class, driving_exp, sex_of_casualty] ).reshape(1,-1)

        pred = get_prediction(data=data1, model=model)


        st.write(f'The predicted severity is: {pred[0]} ')

if __name__ =='__main__':
    main()





