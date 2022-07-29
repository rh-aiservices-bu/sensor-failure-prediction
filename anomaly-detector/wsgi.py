from flask import Flask, render_template, Response, request

# TODO: use dash instead of flask
#from dash import dash, html, dcc
#from dash.dependencies import Input, Output

import json
from managers.preprocess_data_manager import PreprocessDataManager
import pandas as pd

#reloa
app = Flask(__name__)

# 'application' reference required for wgsi / gunicorn
# https://docs.openshift.com/container-platform/3.11/using_images/s2i_images/python.html#using-images-python-configuration
application = app

@app.route('/')
def main():
    return render_template('main.html', col_list=['sensor_25', 'sensor_11', 'sensor_36', 'sensor_34'], size_list=[80,70,60,50,40,30], stds_list=[8,7,6,5,4])


@app.route('/generateData', methods=['GET', 'POST'])
def generate_data():
    #  col_name, points_group_size, regress_group_size, anomaly_std_factor are all obtained
    # from the form in the user interface.
    regression_group_size = int(request.form.get('size_list', '80'))
    points_group_size = 1
    #col_name = 'sensor_34'
    col_name = request.form.get('sensor_list', 'sensor_25')
    anomaly_std_factor = int(request.form.get('stds_list', '4'))

    pdm = PreprocessDataManager(regression_group_size,
                                points_group_size, col_name, anomaly_std_factor,
                                csv_file_name='', use_csv=False, use_postgres=True)
    pdm.process_point()
    return Response(pdm.process_point(), mimetype='text/event-stream')



@app.route('/generateDataFromCsv', methods=['GET', 'POST'])
def generate_data_from_csv():
    #  col_name, points_group_size, regress_group_size, anomaly_std_factor are all obtained
    # from the form in the user interface.
    regression_group_size = int(request.form.get('size_list', '80'))
    points_group_size = 1
    col_name = request.form.get('col_name', '')
    anomaly_std_factor = int(request.form.get('stds_list', '4'))
    uploaded_file = request.files['csv_input']
    if uploaded_file.filename != '':
        uploaded_file.save(uploaded_file.filename)
        file_name = uploaded_file.filename
    else:
        file_name = 'static/casing1.csv'
    
    pdm = PreprocessDataManager(regression_group_size,
                                points_group_size, col_name, anomaly_std_factor,
                                csv_file_name=file_name, use_csv=True, use_postgres=False)
    pdm.process_point()
    return Response(pdm.process_point(), mimetype='text/event-stream')



@app.route('/generateDataFromSynthesis', methods=['GET', 'POST'])
def generate_data_from_synthesis():
    #  col_name, points_group_size, regress_group_size, anomaly_std_factor are all obtained
    # from the form in the user interface.
    regression_group_size = int(request.form.get('size_list', '80'))
    points_group_size = 1
    col_name = request.form.get('col_name', '')
    anomaly_std_factor = int(request.form.get('stds_list', '4'))
    
    pdm = PreprocessDataManager(regression_group_size,
                                points_group_size, col_name, anomaly_std_factor,
                                csv_file_name='', use_csv=False, use_postgres=False)
    pdm.process_point()
    return Response(pdm.process_point(), mimetype='text/event-stream')

if __name__ == "__main__":
    app.run(debug=True, port=8080, host="0.0.0.0")  # nosec

# run gunicorn manually
# TODO: move to readme
# gunicorn wsgi:application -b 0.0.0.0:8080
