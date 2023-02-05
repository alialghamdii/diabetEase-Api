
def predict(filename):
    df = pd.read_csv(filename, encoding='latin-1', header = None, names = columns)
    prediction_data = df.to_csv("prediction_data.csv", encoding='utf-8', index=False)
    return prediction_data