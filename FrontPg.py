import streamlit as st
import numpy as np
import pandas as pd
from datetime import datetime
import os
from src.XGBoost.preprocess import basic_clean
from src.XGBoost.feature_engineering import Feature_Engineering
from Predict import Predict_Ensemble,Predict_FTT

st.title("Space Traffic Management , Find Collision Probability and Risk Factor", text_alignment="center")

test_sample = pd.DataFrame
Mcol1, Mcol2 = st.columns(2)
with Mcol1:
    st.header("Enter CDM Data")
    with st.expander("Enter CDM Data", expanded=True):
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
        # Try parsing safely
        creationTsOfCDM = pd.to_datetime(
        creationTsOfCDM,
        format="%Y-%m-%d %H:%M:%S",
        errors="coerce"   # invalid formats become NaT
        )

        if pd.isna(creationTsOfCDM):
            st.warning("⚠️ Please enter a valid datetime in format YYYY-MM-DD HH:MM:SS")
            # Optionally set a default
        cdmTca = pd.Timestamp.now()
        cdmTca = pd.to_datetime(
        cdmTca,
        format="%Y-%m-%d %H:%M:%S",
        errors="coerce"   # invalid formats become NaT
        )

        if pd.isna(cdmTca):
            st.warning("⚠️ Please enter a valid datetime in format YYYY-MM-DD HH:MM:SS")
            # Optionally set a default
        cdmTca = pd.Timestamp.now()
        
        if submitted and not (pd.isna(creationTsOfCDM) or pd.isna(cdmTca)):
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
            test_sample = Featured_D
            st.write(Featured_D)

with Mcol2:
    st.title("Enter Data for Prediction: ")
    
    if not test_sample.empty:
        st.title("Ensemble Prediction: ")
        Ensemble_Predict = Predict_Ensemble(test_sample)
        if Ensemble_Predict == 1:
            st.write(":red[***HighRisk***]")
        else:
            st.write(":green[***LowRisk***]")
            
        HighR = Predict_FTT(test_sample)
        st.title("FTT Transformer Prediction: ")
        # Get the first prediction label (adjust if you expect multiple rows)
        risk = HighR['risk_label'].iloc[0] if len(HighR) > 0 else None

        if risk == "HighRisk":
            st.markdown("<span style='color:red;font-weight:bold'>HighRisk</span>", unsafe_allow_html=True)
        elif risk == "LowRisk":
            st.markdown("<span style='color:green;font-weight:bold'>LowRisk</span>", unsafe_allow_html=True)
        else:
            # Fallback: print whatever label is present
            st.write(risk)


    
file_path = os.path.join(os.path.dirname(__file__), "data", "sample_Featured_DATA.csv")


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




