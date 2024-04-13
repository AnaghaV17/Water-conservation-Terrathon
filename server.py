from flask import Flask,request, url_for, redirect, render_template, request
import pickle
import numpy as np
import pandas as pd


app = Flask(__name__,static_url_path='/Users/niranjan/Downloads/test/static')

model=pickle.load(open('model.pkl','rb'))


@app.route('/',methods=['POST','GET'])
def hello_world():
    return render_template("index.html")


@app.route('/entry',methods=['POST','GET'])
def entry():
    
    if request.method=='POST':
        parameter_values1=[]
        parameter_values2=[]
        for index, (parameter_name, parameter_value) in enumerate(request.form.items()):
            try:
                parameter_value = int(parameter_value)
                if index < 7:
                    parameter_values1.append(parameter_value)
                else:
                    parameter_values2.append(parameter_value)
            except ValueError:
                pass
        #parameter_values1.pop(0)
        parameter_values2.insert(0,parameter_values1[5])
        if len(parameter_values2 or parameter_values1) == 0:
             return {'error': 'No form values provided or conversion to integers failed.'}


    # try:
        
    #     # Get form values from the request
    #     # form_values = request.form.values()
    #     # form_value_list= list(form_values)



    #     # Convert form values to integers
    #     #int_features = {key: int(value) for key, value in form_values.items()}
    #     #print("Hey! ",int_features)
    #     # Check if int_features is empty
    #     if len(form_value_list) == 0:
    #         return {'error': 'No form values provided or conversion to integers failed.'}
        
    #     # Convert int_features dictionary to a list (if needed)
        
        
    #     # Convert list to a NumPy array
        final1 = np.array(parameter_values1).reshape(1, -1)
        final2 = np.array(parameter_values2).reshape(1, -1)
        
        # Load the machine learning model
        with open("model.pkl", "rb") as file1:
            model1 = pickle.load(file1)

         # Load the machine learning model
        with open("model2.pkl", "rb") as file2:
            model2 = pickle.load(file2)

        # Make predictions using the model
        predictions1 = model1.predict(final1)
        predictions2 = model2.predict(final2)
        
        # Convert predictions to a list (if needed)
        predictions_list1 = predictions1.tolist()
        predictions_list2 = predictions2.tolist()

        if predictions_list1[0]>100:
            algal="ALGAL BLOOM PREDICTED!\n"
        
        else:
            algal="NO ALGAL BLOOM DETECTED\n"

        if predictions_list2==0:
            potable="NOT POTABLE\n"

        else:
            potable="POTABLE\n"
        '''
        parameter=[]
        parameter_val=[]
        for (parameter_name, para_value) in enumerate(request.form.items()):
            para_value = int(para_value)
            parameter_val.append(para_value)
        '''
        #report = pred1+pred2
        # Return predictions as a JSON response
        #return {'predictions algal bloom': predictions_list1[0], 'predictions potability':predictions_list2[0]}



        sum = (parameter_values2[3]*5+parameter_values2[0]*3+parameter_values2[5]*2+parameter_values2[8]*2+parameter_values2[6]*1+parameter_values2[4]*1+parameter_values2[1]*1)/100

        if sum<14:
            wqi='BAD: Severe water quality problems'
        elif sum>15 and sum<39:
            wqi='POOR. Water quality issues requiring attention'
        elif sum>40 and sum<69:
            wqi='GOOD. Some minor deviations, may not be a major concern'
        elif sum>70:
            wqi='VERY GOOD. Minimal deviations from ideal levels'

        return render_template('report.html',wqi=wqi, algal=algal,potable=potable,light=parameter_values1[0],nitrate=parameter_values1[1],iron=parameter_values1[2],phosphate=parameter_values1[3],temperature=parameter_values1[4],ph=parameter_values1[5],co2=parameter_values1[6],hardness=parameter_values2[1],solids=parameter_values2[2],chloramines=parameter_values2[3],sulfate=parameter_values2[4],conductivity=parameter_values2[5],oc=parameter_values2[6],trihalome=parameter_values2[7],turbidity=parameter_values2[8])

    # except Exception as e:
    #     return {'error': str(e)}
    '''try:
        # Retrieve data from the endpoint
        int_features = [int(x) for x in request.form.values()]
        print('This is list: ',int_features)
        # Parse data into a NumPy array
        final = np.array(int_features).reshape(1, 7)
        
        # Load the machine learning model
        with open("model.pkl", "rb") as file:
            model = pickle.load(file)

        # Make predictions using the model
        predictions = model.predict(final)
        
        # Convert predictions to a list (if needed)
        predictions_list = predictions.tolist()
        
        # Return predictions as a JSON response
        return {'predictions': predictions_list}

    except Exception as e:
        return {'error': str(e)}'''
'''def predict():
    int_features=[int(x) for x in request.form.values()]
    final1=[np.array(int_features)]
    final = final1[1:]
    with open("model.pkl", "rb") as file:
        model = pickle.load(file)

    # Reshape final array (if it's a 1D array)
    final = np.array(final)
    print('This is ',final.shape)
    return "hi"
    '''
    #if len(final.shape) == 1:  # Check if it's 1D
        #final = final.reshape(1, -1)  # Reshape to 2D with one row

    # Prediction
    #predictions = model.predict(final)

# Print the predictions
    #print("Predictions:", predictions)
'''df= pd.DataFrame(final)
    df.to_csv("data1.csv")
    df.to_pickle("picky.pkl")
    with open('picky.pkl', 'rb') as f:
        model = pickle.load(f)
    new= pd.read_csv("data1.csv")
    predictions = model.predict(new)
    print(predictions)'''
    #print(int_features)
    #print(final)
    #final=final.reshape(-1, 1)

    # Step 1: Read data from the CSV file
#csv_file_path = "data1.csv"
#data_from_csv = pd.read_csv(csv_file_path)

# Step 2: Process and manipulate the data if necessary
# (You can skip this step if no processing is needed)

# Step 3: Save the processed data into a Python data structure
#testing_data = data_from_csv.values  # Assuming you want to store data as numpy array
#print(testing_data)
# Step 4: Write the data structure to a pickle file
#pkl_file_path = "testing_data.pkl"
#with open(pkl_file_path, 'wb') as pkl_file:
#   pickle.dump(testing_data, pkl_file)

# Step 5: Load the trained model
#model_file_path = "model.pkl"
#with open(model_file_path, 'rb') as model_file:
 #   model = pickle.load(model_file)

# Step 6: Load the testing data
#with open(pkl_file_path, 'rb') as pkl_file:
 #   testing_data = pickle.load(pkl_file)

# Step 7: Make predictions using the loaded model
#predictions = model.predict(testing_data)

# Step 8: Optionally, return or use the predictions
#print(predictions)  # You can replace this with whatever you want to do with the predictions

'''
prediction=model.predict(final)
output='{0:.{1}f}'.format(prediction[0][1], 2)
print(prediction)
if prediction>str(200):
    return render_template('report.html',pred='Your Forest is in Danger.\nProbability of fire occuring is {}'.format(output),bhai="kuch karna hain iska ab?")
else:
    return render_template('report.html',pred='Your Forest is safe.\n Probability of fire occuring is {}'.format(output),bhai="Your Forest is Safe for now")

'''


if __name__ == '__main__':
    app.run(debug=True)