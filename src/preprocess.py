import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder

DATA_DIR = "data"
RAW_FILES = [
    "CZ_6A_Events_2024-08-06.xlsx",
    "CZ_6A_Events_2024-09-06.xlsx",
    "CZ_6A_Events_2024-10-06.xlsx",
    # "CZ_6A_Events_2025-02-06.xlsx",
    # "CZ_6A_Events_2025-06-06.xlsx",
    # "CZ_6A_Events_2025-08-06.xlsx",
]

OUTPUT_DIR = "Merged_DATA.xlsx"

def load_and_merge(data_dir : str , file_list:list[str]) -> pd.DataFrame:
    dfs = []
    for fname in file_list:
        path = os.path.join(data_dir , fname)
        print(f"Loading: {path}")
        df = pd.read_excel(path)
        dfs.append(df)
    merged = pd.concat(dfs, ignore_index=True)
    print("\nMerged Shape : " , merged.shape , "\n")
    return merged


def basic_clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply very basic cleaning:
    - Standardize column names
    - Convert timestamps
    - Fill a few obvious missing values
    (We keep it simple here; more cleaning will come later.)
    """
    # 1) Basic imputations
    df['cdmPc'] = df['cdmPc'].fillna(df['cdmPc'].median())
    df['rso2_objectType'] = df['rso2_objectType'].fillna("DEBRIS")

    df.columns = [c.strip() for c in df.columns]
    rename_map = {
        'condition_24H<tca<72H': 'condition_24H_tca_72H',
        'condition_Radial<100m': 'condition_Radial_100m',
        'condition_InTrack<500m': 'condition_InTrack_500m',
        'condition_CrossTrack<500m': 'condition_CrossTrack_500m',
        'condition_sat2posUnc>1km': 'condition_sat2posUnc_1km',
        'condition_sat2Obs<25': 'condition_sat2Obs_25'
    }

    df = df.rename(columns={k: v for k,v in rename_map.items() if k in df.columns })
   

    # Create hours_to_tca if timestamps exist
    if 'creationTsOfCDM' in df.columns and 'cdmTca' in df.columns:
        df['hours_to_tca'] = (df['cdmTca'] - df['creationTsOfCDM']).dt.total_seconds() / 3600.0
    else:
        df['hours_to_tca'] = None


    # Fill some obvious missing values (Pure Assumption)
    if 'cdmPc' in df.columns:
        df['cdmPc'] = df['cdmPc'].fillna(df['cdmPc'].median())
    if 'rso2_objectType' in df.columns:
        df['rso2_objectType'] = df['rso2_objectType'].fillna("DEBRIS")
    
    # # 3)  categorical features Conversion to category 
    # categorical_cols = [
    #     'SAT1_CDM_TYPE', 'SAT2_CDM_TYPE',
    #     'rso1_objectType', 'rso2_objectType',
    #     'org1_displayName', 'org2_displayName'
    # ]
    # df[categorical_cols] = df[categorical_cols].astype("category")
    
    
    # 5) Target: HighRisk
    df['HighRisk'] = (
        (df['cdmPc'] > 1e-6) & (df['cdmMissDistance'] < 2000)
    ).astype(int)
    print("\nAfter basic_clean:")
    print(df.info())
    return df


def save_clean_data(df : pd.DataFrame , data_dir:str, filename:str)-> str:
    # save Merge and Clean data
    output_path = os.path.join(data_dir , filename)
    print(f"\n Saving Cleanding data to :{output_path}")
    print(df.info())
    df.to_excel(output_path, index=False)
    return output_path


def main():
    #1 Load and merge

    merged_data = load_and_merge(DATA_DIR , RAW_FILES)

    # Clean data
    clean_data = basic_clean(merged_data)

    # save Final Data
    save_clean_data(clean_data , DATA_DIR , OUTPUT_DIR)


# - if you run python src/preprocess.py, it will execute main().

if __name__ == "__main__":
    main()