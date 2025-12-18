import streamlit as st
import numpy as np
import pandas as pd
from datetime import datetime
import os
from src.XGBoost.preprocess import basic_clean
from src.XGBoost.feature_engineering import Feature_Engineering


st.title("Space Traffic Management , Find Collision Probability and Risk Factor")
st.header("Space Traffic Management")



# st.title("CDM Input Form")

# example_data = {
#     "cdmMissDistance": 597,
#     "cdmPc": 1.092937e-06,
#     "creationTsOfCDM": "2024-09-05 13:15:44",
#     "SAT1_CDM_TYPE": "EPHEM",
#     "SAT2_CDM_TYPE": "HAC",
#     "cdmTca": "2024-09-06 10:57:18.924",
#     "rso1_noradId": 55772,
#     "rso1_objectType": "PAYLOAD",
#     "org1_displayName": "SpaceX",
#     "rso2_noradId": 55876,
#     "rso2_objectType": "DEBRIS",
#     "org2_displayName": "NONE",
#     "condition_cdmType=EPHEM:HAC": True,
#     "condition_24H_tca_72H": False,
#     "condition_Pc>1e-6": True,
#     "condition_missDistance<2000m": True,
#     "condition_Radial_100m": False,
#     "condition_Radial<50m": False,
#     "condition_InTrack_500m": True,
#     "condition_CrossTrack_500m": True,
#     "condition_sat2posUnc_1km": True,
#     "condition_sat2Obs_25": True,
#     "hours_to_tca": 21.693034444444443,
#     "HighRisk": 1,
#     "log_cdmPc": 1.0929364027447921e-06,
#     "inv_miss_distance": 0.0016722408026755853,
#     "tca_bin": 1,
#     "same_sat_type": 0,
#     "is_debris_pair": 0,
#     "close_all_axes": 0,
#     "risky_uncertainty": 1,
#     "distance_ratio": 26.307632038436076,
#     "object_type_match": 0
# }

# st.dataframe(pd.DataFrame([example_data]))

file_path = os.path.join(os.path.dirname(__file__), "data", "sample_Featured_DATA.csv")
st.write("Current working directory:", os.getcwd())
st.write("Files:", os.listdir())
st.write(os.listdir("data"))

sample_df = pd.read_csv(file_path)
    
with st.expander('Sample Data'):
    st.write("Shape of Sample Data :",sample_df.shape)
    st.write(sample_df)
with st.expander("About Data"):
    st.write("Orginal data Shape : :green[***(574289,33)***]")
    HeatMap_Img_Path = os.path.join(os.path.dirname(__file__), "outputs", "Plots" ,"Correlation_Heatmap_Top.png" )
    st.image(HeatMap_Img_Path)

with st.sidebar:
    st.header("About app")
    st.write("Created By Nexus IIT-J")




with st.expander("Enter Manual Input"):
    with st.form("cdm_form"):
        col1, col2 = st.columns(2)
        with col1:
            cdmMissDistance = st.number_input("cdmMissDistance (Numeric)", placeholder="777")
            SAT1_CDM_TYPE = st.selectbox("SAT1_CDM_TYPE", ["EPHEM", "HAC", "OTHER"])
            creationTsOfCDM = st.text_input("creationTsOfCDM (YYYY-MM-DD HH:MM:SS)", placeholder="2024-09-05 13:15:44", value="2024-09-05 13:15:44")
            rso1_noradId = st.number_input("rso1_noradId", value=0)
            rso1_objectType = st.text_input("rso1_objectType", value="DEBRIS")
            org1_displayName = st.text_input("org1_displayName",value="NONE")
            st.write("Below all Condtion are either True / False ")
            condition_cdmType = st.selectbox("condition_cdmType=EPHEM:HAC",["True","False"])
            condition_24H_tca_72H = st.selectbox("condition_24H_tca_72H",["True","False"])
            condition_Pc = st.selectbox("condition_Pc>1e-6",["True","False"])
            condition_missDistance = st.selectbox("condition_missDistance<2000m",["True","False"])
            condition_Radial_100m = st.selectbox("condition_Radial_100m",["True","False"])
        
        with col2:
            cdmPc = st.number_input("cdmPc", placeholder="0.00000925", format="%.10f")
            SAT2_CDM_TYPE = st.selectbox("SAT2_CDM_TYPE", ["EPHEM", "HAC", "OTHER"])
            cdmTca = st.text_input("cdmTca (YYYY-MM-DD HH:MM:SS.sss)",value="2024-09-06 10:57:18.924")
            rso2_noradId = st.number_input("rso2_noradId", value=0)
            rso2_objectType = st.text_input("rso2_objectType",value="DEBRIS")
            org2_displayName = st.text_input("org2_displayName",value="NONE")
        # Boolean conditions
            st.write(" | | ")
            condition_Radial_50m = st.selectbox("condition_Radial<50m",["True","False"])
            condition_InTrack_500m = st.selectbox("condition_InTrack_500m",["True","False"])
            condition_CrossTrack_500m = st.selectbox("condition_CrossTrack_500m",["True","False"])
            condition_sat2posUnc_1km = st.selectbox("condition_sat2posUnc_1km",["True","False"])
            condition_sat2Obs_25 = st.selectbox("condition_sat2Obs_25",["True","False"])

        
        
        submitted = st.form_submit_button("Submit")

    if submitted:
        new_row = {
            "cdmMissDistance": cdmMissDistance,
            "cdmPc": cdmPc,
            "creationTsOfCDM": creationTsOfCDM,
            "SAT1_CDM_TYPE": SAT1_CDM_TYPE,
            "SAT2_CDM_TYPE": SAT2_CDM_TYPE,
            "cdmTca": cdmTca,
            "rso1_noradId": rso1_noradId or 0,
            "rso1_objectType": rso1_objectType,
            "org1_displayName": org1_displayName,
            "rso2_noradId": rso2_noradId,
            "rso2_objectType": rso2_objectType,
            "org2_displayName": org2_displayName,
            "condition_cdmType=EPHEM:HAC": condition_cdmType,
            "condition_24H_tca_72H": condition_24H_tca_72H,
            "condition_Pc>1e-6": condition_Pc,
            "condition_missDistance<2000m": condition_missDistance,
            "condition_Radial_100m": condition_Radial_100m,
            "condition_Radial<50m": condition_Radial_50m,
            "condition_InTrack_500m": condition_InTrack_500m,
            "condition_CrossTrack_500m": condition_CrossTrack_500m,
            "condition_sat2posUnc_1km": condition_sat2posUnc_1km,
            "condition_sat2Obs_25": condition_sat2Obs_25,
        }
        for key, value in new_row.items():
            if value == "True":
                new_row[key] = 1
            elif value == "False":
                new_row[key] = 0
        
        st.write("### Submitted Data")
        Input_data= pd.DataFrame([new_row])
        preprocess_D = basic_clean(Input_data)
        Featured_D = Feature_Engineering(preprocess_D)
        st.write(Featured_D)



