import pandas as pd
import time
import os
from sklearn.preprocessing import Normalizer
import numpy as np
import tensorflow as tf

#file="C:\\Users\\USER\\Desktop\\Machine Learning\\bot_dataset.csv"
file="E:\\cicflowmeter\\flows.csv"
row_num = 0
#columns = ['dst_port', 'fwd_iat_mean', 'fwd_iat_max', 'flow_iat_mean','fwd_pkts_s', 'flow_pkts_s', 'fwd_iat_tot', 'init_fwd_win_byts','flow_duration', 'bwd_pkt_len_mean', 'bwd_seg_size_avg', 'pkt_size_avg','fwd_iat_min', 'flow_iat_max', 'flow_iat_min', 'bwd_pkts_s','subflow_fwd_pkts', 'fwd_pkt_len_mean', 'fwd_pkt_len_std'] 
#model=tf.keras.models.load_model("E:\\project\\NIDS\\NidsApp\\model18.h5")
columns=['dst_port', 'init_fwd_win_byts', 'fwd_seg_size_min', 'flow_iat_min', 'fwd_iat_mean', 'flow_iat_max', 'flow_pkts_s', 'fwd_iat_tot', 'fwd_pkts_s', 'pkt_size_avg', 'flow_iat_mean', 'fwd_iat_min', 'fwd_iat_max', 'fwd_pkt_len_max', 'fwd_header_len', 'totlen_fwd_pkts', 'pkt_len_max', 'fwd_pkt_len_mean', 'bwd_seg_size_avg', 'subflow_fwd_byts', 'pkt_len_mean', 'bwd_pkts_s', 'flow_duration', 'flow_byts_s', 'fwd_seg_size_avg', 'bwd_pkt_len_mean', 'init_bwd_win_byts', 'subflow_bwd_pkts']
#columns=['Dst Port','Init Fwd Win Byts','Fwd Seg Size Min','Flow IAT Min','Fwd IAT Mean','Flow IAT Max','Flow Pkts/s','Fwd IAT Tot','Fwd Pkts/s','Pkt Size Avg','Flow IAT Mean','Fwd IAT Min','Fwd IAT Max','Fwd Pkt Len Max','Fwd Header Len','TotLen Fwd Pkts','Pkt Len Max','Fwd Pkt Len Mean','Bwd Seg Size Avg','Subflow Fwd Byts','Pkt Len Mean','Bwd Pkts/s','Flow Duration','Flow Byts/s','Fwd Seg Size Avg','Bwd Pkt Len Mean','Init Bwd Win Byts','Subflow Bwd Pkts']
model=tf.keras.models.load_model("C:\\Users\\USER\\Desktop\\Machine Learning\\NIDS_model.h5")
result_file = 'results.csv'
state_file = 'state.txt'

# check if the state file exists and read the last row number
if os.path.exists(state_file):
    with open(state_file, 'r') as f:
        row_num = int(f.read())

while True:
    # Read the current file
    df = pd.read_csv(file)

    # Check if the number of rows has changed
    if len(df) > row_num:
        # Read the new data with the header
        new_df = df.iloc[row_num:]
        row_num = len(df)

        # Use the new data to make predictions with model
        selected_df = new_df[columns]
        #df=selected_df.iloc[:, :-1].values
        df=selected_df
        scaler = Normalizer().fit(df)
        df = scaler.transform(df)
        np.set_printoptions(precision=3)
        df = df.reshape(len(df), df.shape[1], 1)
        df = np.reshape(df, (df.shape[0], 1, df.shape[1]))
        output=model.predict(df)

        # get the predicted class for each row
        predicted_class = np.argmax(output, axis=1)
        predicted_class = np.where(predicted_class == 0, 'normal', 'abnormal')


        # add the predicted class as a new column in the original DataFrame
        selected_df['Prediction'] = predicted_class

        # store the results in a new CSV file
        #selected_df.to_csv(result_file, mode='a', header=not os.path.exists(result_file), index=False)
        if not os.path.exists(result_file):
            # Write the header if the file does not exist
            selected_df.to_csv(result_file, index=False)
        else:
            # Overwrite the data while maintaining the header
            with open(result_file, 'r') as f:
                header = f.readline()
            selected_df.to_csv(result_file, mode='w', header=False, index=False)
            with open(result_file, 'r+') as f:
                content = f.read()
                f.seek(0, 0)
                f.write(header.rstrip('\r\n') + '\n' + content)


        

        print(predicted_class)
        # filter the results to include only the rows with normal predictions
        #normal_results = results[results['prediction'] == 'normal']

        

        # store the last row number in the state file
        with open(state_file, 'w') as f:
            f.write(str(row_num))

    # Wait for the specified interval
    time.sleep(5)
    