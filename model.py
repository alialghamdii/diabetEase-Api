
def predict(filename):
    prediction_data = filename.to_csv("prediction_data.csv", encoding='utf-8', index=False)
    return prediction_data