import numpy as np
import pandas as pd

def missing_to_nan(df, feat_info):
    missing_values = ['X', 'XX','-1','0', '9'] 
    columns = feat_info['attribute'] 
    list_missing_values_array = []

    for missing_value_array in feat_info['missing_or_unknown']: 
        new_missing_array = [] 
        for i in missing_values: 
            if i in missing_value_array and i not in ['X', 'XX']: 
                new_missing_array.append(int(i))
            elif i in missing_value_array and i in ['X', 'XX']:
                new_missing_array.append(i)
        list_missing_values_array.append(new_missing_array)


    for index, row in feat_info.iterrows():
        df[row["attribute"]].replace(list_missing_values_array[index], 
                                         [np.NaN]*len(list_missing_values_array[index]), 
                                         inplace=True)
    return df


def remove_columns(df, feat_info):
    """
    Remove columns with unusually high missing values.
    
    Arguments: 
        df: A dataframe of Demographics data. 
        feat_info: A dataframe of summary of feature attributes for demographics data; 
                   85 features (rows) x 4 columns
        
    Returns:
        df_count: A dataframe with columns containing unusually high missing values dropped. 
    """
    missingvalues = dict()
    for feature in feat_info["attribute"]:
        missingvalues[feature] = df[feature].isnull().sum()
    df_count = pd.DataFrame(list(missingvalues.items()))
    df_count.columns =["Features","Count"]

    # sort df by Count column
    df_count = df_count.sort_values(by=["Count"], ascending=False).reset_index(drop=True)
    
    df_dropped = df.drop(labels=df_count["Features"][:6].values, axis=1)
    return df_dropped

def remove_rows(df, n):
    """
    Removes rows with more than n number of columns with missing values.
    
    Arguments:
        df: A dataframe of demographics data.
        n(int): The threshold number of columns which can have missing values, above which that row will be dropped.
    Returns:
        df: A dataframe with rows with at the most n missing values.   
    """
    row_missing_values = df.isnull().sum(axis=1)

    df_row_mv = pd.DataFrame(list(row_missing_values.values))
    df_row_mv.columns =["Count_Row"]

    # sort df by Count column
    df_row_mv = df_row_mv.sort_values(by=["Count_Row"], ascending=False)
    
    df_dropped = df.drop(df_row_mv[df_row_mv["Count_Row"]>n].index)
    print("{} rows dropped.".format(df.shape[0] - df_dropped.shape[0]))
    return df_dropped

def process_PJ(df):
    
    # interval-type variable for various decades
    # 40s = 0, 50s = 1, 60s = 2, 70s = 3, 80s = 4, 90s = 5
    decade = list()
    # binary-type variable for Mainstream = 0, Avantgarde = 1
    movement = list()

    mainstream = [1,3,5,8,10,12,14]
    avantgarde = [2,4,6,7,9,11,13,15]
    decade_dict = {1:0,2:0,3:1,4:1,5:2,6:2,7:2,8:3,9:3,10:4,11:4,12:4,13:4,14:5,15:5}

    data = df["PRAEGENDE_JUGENDJAHRE"]
    for value in data:
        if value in mainstream and value in decade_dict.keys():
            movement.append(0)
            decade.append(decade_dict[value])
        elif value in avantgarde and value in decade_dict.keys():
            movement.append(1)
            decade.append(decade_dict[value])
        else:
            movement.append(np.NaN)
            decade.append(np.NaN)
    df = df.drop(["PRAEGENDE_JUGENDJAHRE"], axis=1)
    df["MOVEMENT_PRAEGENDE_JUGENDJAHRE"] = movement
    df["DECADE_PRAEGENDE_JUGENDJAHRE"] = decade
    return df

def process_CI2015(df): 
    """
    For Wealth, the variables are - 
    Wealthy Households = 1, Prosperous Households = 2, Comfortable Households = 3, Less Affluent Households = 4, 
    Poorer Households = 5

    For Life Stage, the variables are - 
    Pre-Family Couples & Singles = 1, Young Couples With Children = 2, Families With School Age Children = 3,
    Older Families &  Mature Couples = 4, Elders In Retirement = 5
    """
    wealth = list()
    life_stage = list()
    data = df["CAMEO_INTL_2015"]
    for value in data:
        if value is not np.NaN: 
            wealth.append(int(int(value)/10))
            life_stage.append(int(value)%10)
        else:
            wealth.append(np.NaN)
            life_stage.append(np.NaN)
    df = df.drop(["CAMEO_INTL_2015"], axis=1)
    df["WEALTH_CAMEO_INTL_2015"] = wealth
    df["LIFESTAGE_CAMEO_INTL_2015"] = life_stage
    return df

def reencode_categorical_features(df, feat_info):
    """
    Drops multi-level categorical variables and creates dummy variables for binary-level cateogrical variables.
    """
    cleaned_features = df.columns.values
    feat_info_cleaned = feat_info[feat_info['attribute'].isin(cleaned_features)]
    
    # A list of categorical variables
    categoricals = list(feat_info_cleaned[feat_info_cleaned['type'] == 'categorical']['attribute'].values)
    
    multi_categorical = list()
    binary_categorical = list()
    for category in categoricals:
        if len(df[category].value_counts()) > 2:
            multi_categorical.append(category)
        else:
            binary_categorical.append(category)
            
    binary_categorical.remove("GREEN_AVANTGARDE")
    
    # Drop the multi-level categorical variables
    df.drop(multi_categorical, axis=1, inplace=True)
    
    # Create dummy variables for 2-level categorical variables
    # Drop the original feature and concatenate the corresponding variables to the dataframe
    for feature in binary_categorical:
        df = pd.concat([df.drop([feature], axis=1), 
                        pd.get_dummies(df[feature], prefix=feature)], 
                        axis=1)
    return df


def clean_data(df, feat_info, n):
    """
    Perform feature trimming, re-encoding, and engineering for demographics
    data
    
    INPUT: Demographics DataFrame
    OUTPUT: Trimmed and cleaned demographics DataFrame
    """
    
    # Put in code here to execute all main cleaning steps:
    # convert missing value codes into NaNs, ...
    df = missing_to_nan(df, feat_info)

    # Drop the columns comprising the highest missing values
    df = remove_columns(df, feat_info)
    
    # Drop the rows with more than 20 missing values
    df = remove_rows(df, n)
    
    # Re-encode categorical features
    df = reencode_categorical_features(df, feat_info)
    
    # Re-encode mixed features
    df = process_PJ(df)
    df = process_CI2015(df)
    
    # Remove unnecessary mixed features
    df.drop(["LP_LEBENSPHASE_FEIN", "LP_LEBENSPHASE_GROB"], axis=1, inplace=True)
    
    # Return the cleaned dataframe.
    return df