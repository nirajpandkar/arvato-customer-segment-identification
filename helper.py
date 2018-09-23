def missing_to_nan(df):
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


