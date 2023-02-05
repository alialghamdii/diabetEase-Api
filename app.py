from flask import Flask, request
import os
import csv
import jsonpickle
from model import predict

app = Flask(__name__)

@app.route("/")
def index():
    return "Welcome to the API."

@app.route("/send_csv", methods=["POST"])
def send_csv():
    file = request.files['file']
    file.save(file.filename)
    return "File received and saved successfully."

@app.route("/get_csv", methods=["GET"])
def get_csv():
    csv_files = os.listdir("csv_files")
    if len(csv_files) == 0:
        return "No files found."
    else:
        latest_file = max(csv_files, key=lambda x: os.path.getctime("csv_files/" + x))
        with open("csv_files/" + latest_file) as f:
            reader = csv.reader(f)
            data = [row for row in reader]
            predictions = predict(latest_file)

if __name__ == "__main__":
    app.run(debug=True)
