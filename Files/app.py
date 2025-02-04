import gradio as gr
import numpy as np 
import joblib

model = joblib.load("model4.joblib")

def myfunc(Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age):
    data = np.array([[Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age]])
    p = model.predict_proba(data)
    return p[0][1]

demo = gr.Interface(fn=myfunc,
                   inputs=[
                       gr.Number(label="Pregnancies"),
                       gr.Number(label="Glucose"),
                       gr.Number(label="Blood Pressure"),
                       gr.Number(label="Skin Thickness"),
                       gr.Number(label="Insulin"),
                       gr.Number(label="BMI"),
                       gr.Number(label="Diabetes Pedigree Function"),
                       gr.Number(label="Age"),                       
                   ],
                   outputs=gr.Textbox(label="Probability"),
                   title="Diabetes prediction App",
                   description="Enter patient info to predict diabetes risk")
demo.launch()
    