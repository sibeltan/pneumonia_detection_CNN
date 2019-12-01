from flask import Flask, render_template, request, Response
import numpy as np
import pickle
import os


app = Flask(__name__)

# define the route
@app.route('/')
# create the controller
def home():
   # return the view
   return render_template('index.html', result_Css = '', result_logo = 'twitter-logo', result_message = 'No Emergency')

@app.route('/submit')
# create the controller
def submit():
    user_input = request.args
    tweet_text = user_input['tweet-text']
    data = np.array([tweet_text])


    # model = pickle.load(open(os.getcwd() + '/mysite/assets/model.p', 'rb'))
    model = pickle.load(open(os.getcwd() + '/assets/model.p', 'rb'))
    prediction = model.predict(data)[0]

    # return the view
    if prediction > 0:
        return render_template('index.html', result_Css = 'fire-alert', result_logo = 'twitter-logo-fire', result_message = 'Fire Alert!', tweet_text=tweet_text)
    else:
        return render_template('index.html', result_Css = '', result_logo = 'twitter-logo', result_message = 'No Emergency', tweet_text=tweet_text)

@app.route('/<string:page_name>/')
def render_static(page_name):
    return render_template('%s.html' % page_name)

if __name__ == "__main__":
    app.run(debug=True)
