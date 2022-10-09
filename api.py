from flask import Flask, request, url_for, redirect, render_template
from flask_cors import CORS
import json
import predictor as pr

import os

app = Flask(__name__)
CORS(app)


@app.route('/forecasting', methods=['GET', 'POST'])
def forecasting():
    no_of_months = int(request.form['no_of_months'])

    result = pr.getData(no_of_months)
    output_value = []

    for x in range(len(result)):
        output_value.append(round(result[x]['value'], 1))

    return_str = '{ "next month prediction" : ' + str(output_value[0]) + ' }'

    return json.loads(return_str)


if __name__ == '__main__':
    app.run(host="192.168.1.3", port=8000, debug=True)
