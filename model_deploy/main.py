## Importing the libraries
from fastapi import FastAPI
from pydantic import BaseModel
import json
from pyspark.sql import Row
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import LinearSVC, LinearSVCModel

## Initialize Spark session
spark = SparkSession.builder \
    .appName("DiabetesPrediction") \
    .config("spark.master", "local[*]") \
    .config("spark.hadoop.dfs.client.use.datanode.hostname", "true") \
    .getOrCreate()

## Creating a Fastapi object
app = FastAPI()

## Using Pydantic lib, defining the data type for all the inputs
class model_input(BaseModel):
    
    pregnancies : int
    Glucose : int
    BloodPressure : int
    SkinThickness : int
    Insulin : int
    BMI : float
    DiabetesPedigreeFunction : float
    Age : int       
        
# Load the saved model 
loaded_model = LinearSVCModel.load("diabetes_model.sav")

## Creating a POST request to the API
@app.post('/diabetes_prediction')
def diabetes_predd(input_parameters : model_input):
    
    input_data = input_parameters.json()
    input_dictionary = json.loads(input_data)
    
    preg = input_dictionary['pregnancies']
    glu = input_dictionary['Glucose']
    bp = input_dictionary['BloodPressure']
    skin = input_dictionary['SkinThickness']
    insulin = input_dictionary['Insulin']
    bmi = input_dictionary['BMI']
    dpf = input_dictionary['DiabetesPedigreeFunction']
    age = input_dictionary['Age']
        
    input_list = [preg, glu, bp, skin, insulin, bmi, dpf, age]
    
    # Create a Row object with column names and corresponding values
    input_row = Row(Pregnancies=input_list[0], 
                    Glucose=input_list[1], 
                    BloodPressure=input_list[2], 
                    SkinThickness=input_list[3], 
                    Insulin=input_list[4], 
                    BMI=input_list[5], 
                    DiabetesPedigreeFunction=input_list[6], 
                    Age=input_list[7])

    # Create a DataFrame with the input row
    input_df = spark.createDataFrame([input_row])

    # Assemble the input features into a vector
    feature_columns = ['Pregnancies', 'Glucose','BloodPressure', 'SkinThickness',
                       'Insulin','BMI','DiabetesPedigreeFunction','Age']
    vector_assembler = VectorAssembler(inputCols=feature_columns, outputCol='features')
    input_df = vector_assembler.transform(input_df)

    # Make predictions using the loaded SVM model
    predictions = loaded_model.transform(input_df)

    # Show the predictions
    prediction = predictions.select('prediction').collect()[0][0]
    
    if (prediction == 0):
        return 'The person is not diabetic'
    else:
        return 'The person is diabetic'
        
if __name__ == '__main__':
    uvicorn.run(app, host = '127.0.0.1', port=8000)