from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load the model
model_path = 'xgb_regressor_model.pkl'  # Update this path
with open(model_path, 'rb') as model_file:
    model_pipeline = pickle.load(model_file)


def encode_month(month):
    categories = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
    return [1 if month == category else 0 for category in categories]

def encode_day(day):
    categories = ['mon', 'tue', 'wed', 'thu', 'fri', 'sat', 'sun']
    return [1 if day == category else 0 for category in categories]



@app.route('/')
def home():
    return render_template('index.html')




    
@app.route('/predict', methods=['POST'])
def predict():
    try:
    
        data = request.form.to_dict()
        
    
        encoded_month = encode_month(data['month'])
        encoded_day = encode_day(data['day'])
        
    
        del data['month']
        del data['day']
        
    
        for feature in data.keys():
            data[feature] = [float(data[feature])]
        

        features_df = pd.DataFrame(data)
        

        month_columns = [f'month_{cat}' for cat in ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']]
        day_columns = [f'day_{cat}' for cat in ['mon', 'tue', 'wed', 'thu', 'fri', 'sat', 'sun']]
        
        for i, column_name in enumerate(month_columns + day_columns):
            if i < len(encoded_month):
                features_df[column_name] = encoded_month[i]
            else:
                features_df[column_name] = encoded_day[i - len(encoded_month)]
        
        
        prediction = model_pipeline.predict(features_df)
        
        
        prediction_exp = np.exp(prediction)
        
        print(f'Predicted Burned Area: {prediction_exp[0]:.2f} ha')
        significant_area_threshold = 10.0  
        if prediction_exp > significant_area_threshold:
            message = (f'Emergency! Predicted Burned Area: {prediction_exp[0]:.2f} ha at coordinates X={request.form["X"]}, '
                       f'Y={request.form["Y"]}. Please call 911 immediately.')
        else:
            message = f'Predicted Burned Area: {prediction_exp:.2f} ha.'
    
        return render_template('index.html', prediction_text=message)
    except Exception as e:
        return jsonify({'error': str(e)})


if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0")
