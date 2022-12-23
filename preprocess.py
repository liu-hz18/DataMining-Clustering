import pandas as pd
import numpy as np
import json

DATA_PATH = "./data/diabetic_data.csv"
DATA_SAVE_PATH = "./data/cleaned.csv"
DATA_MAP_PATH = "./data/data_map.json"


if __name__ == '__main__':
    raw_data = pd.read_csv(DATA_PATH)
    print(raw_data.columns)
    print(raw_data.head())
    
    new_data = {}
    data_map = {}

    # 输出每列的取值集合
    for col in raw_data.columns:
        data_map[col] = {}
        col_data = raw_data[col].values.tolist()
        unique_values = np.unique(col_data)
        print(f"{col}: {np.unique(col_data)}")
        unique_values = list(unique_values)
        
        new_num_list = []
        if isinstance(col_data[0], str):
            for data in col_data:
                if data == "?":
                    new_num_list.append(np.nan)
                else:
                    new_num_list.append(unique_values.index(data))
            for i, val in enumerate(unique_values):
                data_map[col][i] = val
        else:
            new_num_list = col_data

        new_data[col] = new_num_list
    
    df = pd.DataFrame(new_data)
    df = df.sort_values(by='encounter_id')
    print(df.head())
    df.to_csv(DATA_SAVE_PATH, sep=",", index=False, header=True)

    print(data_map)
    json.dump(data_map, open(DATA_MAP_PATH, "w"), indent=2)
