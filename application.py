from flask import Flask,request, render_template
import pickle
import pandas as pd

# Create a flask application
application = Flask(__name__)
app = application

#create homepage path
@app.route('/')
def home_page():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def prediction(): 
   if request.method == 'GET':
      return render_template('index.html')
   else:
       with open('D:\Basic Deployment\models\irismodel.pkl','rb') as f:
           model = pickle.load(f)
       
       sepal_length = float(request.form.get('sepal_length'))
       sepal_width = float(request.form.get('sepal_width'))
       petal_length = float(request.form.get('petal_length'))
       petal_width = float(request.form.get('petal_width'))

       df = pd.DataFrame([sepal_length,sepal_width,petal_length,petal_width]).T
       
       df.columns = ['sepal_length','sepal_width','petal_length','petal_width']

       prediction = model.predict(df)
       if prediction == 0:
           prediction = 'Setosa'
       elif prediction == 1:
           prediction = 'Versicolor'
       else:
           prediction = 'Verginica'

       return render_template('index.html',prediction=prediction)

#  Running the app
if __name__ == '__main__':
    #app.run(host='0.0.0.0')
    app.run(host='0.0.0.0', port=8000)