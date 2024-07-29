from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import pickle
import sklearn
import gdown

print(sklearn.__version__)
app = Flask(__name__)

def download_model_from_gdrive(file_id):
    url = f'https://drive.google.com/uc?id={file_id}'
    output = 'model.pkl'
    gdown.download(url, output, quiet=False)
    with open(output, 'rb') as f:
        model = pickle.load(f)
    return model

GDRIVE_FILE_ID = '1Z93F3LgKeplV1qS4Z_FzPMKzeszh6chc'  # Replace with your actual file ID

model = download_model_from_gdrive(GDRIVE_FILE_ID)

@app.route('/')
def hello():
    return render_template('index.html')

@app.route('/prediction', methods=['GET', 'POST'])
def predict():
    print(request.method)
    if request.method == "POST":
        try:
            month = int(request.form['month'])
            age = int(request.form['age'])
            annual_income = float(request.form['annual_income'])
            Num_Bank_Accounts = int(request.form['Num_Bank_Accounts'])
            Num_Credit_Card = int(request.form['Num_Credit_Card'])
            Interest_Rate = float(request.form['Interest_Rate'])
            Num_of_Loan = int(request.form['Num_of_Loan'])
            Delay_from_due_date = int(request.form['Delay_from_due_date'])
            Num_of_Delayed_Payment = int(request.form['Num_of_Delayed_Payment'])
            Changed_Credit_Limit = float(request.form['Changed_Credit_Limit'])
            Num_Credit_Inquiries = int(request.form['Num_Credit_Inquiries'])
            Credit_Mix = request.form['Credit_Mix']
            Outstanding_Debt = float(request.form['Outstanding_Debt'])
            Credit_Utilization_Ratio = float(request.form['Credit_Utilization_Ratio'])
            Credit_History_Age = float(request.form['Credit_History_Age'])
            Total_EMI_per_month = float(request.form['Total_EMI_per_month'])
            Monthly_Balance = float(request.form['Monthly_Balance'])
            Amount_invested_monthly = float(request.form['Amount_invested_monthly'])

            # Calculate additional features
            Debt_Per_Card = Outstanding_Debt / Num_Credit_Card
            Debt_to_Income_Ratio = Outstanding_Debt / annual_income
            Delayed_Payments_Per_Card = Num_of_Delayed_Payment / Num_Credit_Card
            Total_Monthly_Expenses = Total_EMI_per_month + Amount_invested_monthly
            
            pass_values = np.array([
[Outstanding_Debt, Credit_Mix, Interest_Rate, Credit_History_Age, Delay_from_due_date, Changed_Credit_Limit, Monthly_Balance, Credit_Utilization_Ratio, Num_Credit_Inquiries, month, Num_Credit_Card, annual_income, Total_EMI_per_month, Num_of_Delayed_Payment, age, Num_Bank_Accounts, Num_of_Loan, Debt_to_Income_Ratio, Debt_Per_Card, Delayed_Payments_Per_Card, Total_Monthly_Expenses]
            ])

            print(pass_values)
            # Make prediction
            credit_score = model.predict(pass_values)
            print("Credit score is",credit_score[0])
            if credit_score[0]==2:
                credit_cat = "Good"
            elif credit_score[0]==1:
                credit_cat = "Standard"
            else:
                credit_cat = "Bad"
            return render_template('result.html', credit_score=credit_cat)
        
        except Exception as e:
            print(f"Error: {e}")
            return render_template('index.html', error=str(e))
    
    return render_template('index.html')
if __name__ == '__main__':
    app.run()
