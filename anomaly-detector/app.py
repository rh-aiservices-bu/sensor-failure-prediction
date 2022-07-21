from flask import Flask, render_template, Response, request
import json
from graph_manager import GraphManager as GM
from static.data.data_prep import load_sensor
import pandas as pd


app = Flask(__name__)
@app.route('/')
def main():
    return render_template('main.html', col_list=['sensor_25', 'sensor_11', 'sensor_36', 'sensor_34'])

@app.route('/generate-graph', methods=['GET', 'POST'])
def generate_graph():
    chosen_sensor = request.form or '{}'
    df, anomalies = load_sensor(json.loads(chosen_sensor).get('sensor'))
    buffer = GM.plot_data(df, anomalies)
    return buffer

if __name__ == '__main__':
    app.run(port=8080, debug=True,host='0.0.0.0')
