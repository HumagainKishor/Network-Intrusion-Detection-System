from flask import Flask,jsonify
import pandas as pd


app = Flask(__name__)

@app.route('/network-log')
def pred():
    while True:
        #read the csv file 
        df = pd.read_csv('results.csv')

        #convert the pandas dataframe to json format
        data=df.to_json(orient='records')
        return data
        

if __name__ == '__main__':
    app.run(debug=True)
