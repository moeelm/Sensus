from flask import Flask, render_template, request
app = Flask(__name__)
from predict import predict_sentiment
@app.route('/', methods = ['GET', 'POST'])
def home():
    if(request.method == 'GET'):
        return render_template('index.html')
    else:
        data = request.form
        print(request.form)
        return (predict_sentiment(request.form['text']) and 'negative' or 'positive')

    