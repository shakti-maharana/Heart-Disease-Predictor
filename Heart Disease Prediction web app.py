# -- coding: utf-8 --
"""
Created on Sun Jun 29 23:03:07 2025

@author: Shakti
"""

import numpy as np
import pickle
import streamlit as st

# load the saved model
loaded_data = pickle.load(open("trained_model (3).sav",'rb'))
# Extract the model and the scaler
loaded_model = loaded_data['model']
loaded_scaler = loaded_data['scaler']

# created a function for Prediction

def heart_disease_prediction(user_input):
    
    np_input = np.asarray(user_input) # converted to numpy array
    r_input = np_input.reshape(1,-1)
    prediction = loaded_model.predict(r_input)
    print(prediction)
    if (prediction[0]== 0):
      return 'The Person does not have a Heart Disease'
    else:
      return 'The Person has Heart Disease'
  
    
def main():
    
    
    # Giving Title 
    st.title('❤️Heart Disease Prediction')
    # Model Accuracy
    st.markdown("### 🔹 Model Accuracy: ≈ 86%")

    # Feature Table
    st.markdown("### 🧠 Feature Description and Safe Ranges")
    st.markdown("""
    | Feature                     | Description                                      | Safe/Normal Range                   |
    |----------------------------|--------------------------------------------------|-------------------------------------|
    | *Age*                    | Age of the person                                | 18 - 65 years (generally lower risk) |
    | *Gender*                 | 1 = Male, 0 = Female                             | —                                   |
    | *Chest Pain Type (cp)*   | 0–3 categories of chest pain                     | Lower cp values usually safer       |
    | *Resting BP (trestbps)*  | Resting Blood Pressure (mm Hg)                  | 90 - 120 mm Hg                      |
    | *Cholesterol (chol)*     | Serum cholesterol level (mg/dL)                 | Less than 200 mg/dL                 |
    | *Fasting Sugar (fbs)*    | >120 mg/dL → 1; else → 0                         | 0 (normal fasting sugar)            |
    | *Rest ECG (restecg)*     | Resting ECG results (0–2)                       | 0 (normal)                          |
    | *Max Heart Rate (thalach)* | Max heart rate during exercise                | 140 - 190 bpm (depends on age)      |
    | *Exercise Angina (exang)*| 1 = Yes, 0 = No                                  | 0 (no angina)                       |
    | *Oldpeak*                | ST depression induced by exercise                | Lower is better (≈ 0 is ideal)      |
    | *Slope*                  | Slope of the peak exercise ST segment (0–2)      | 2 is normal                         |
    | *CA*                     | No. of major vessels colored by fluoroscopy (0–3)| 0 is healthiest                     |
    | *Thal*                   | 1 = Normal, 2 = Fixed Defect, 3 = Reversible     | 1 (normal)                          |
    """, unsafe_allow_html=True)
        
    age = st.text_input('Age')
    sex = st.text_input('Gender')
    cp = st.text_input('Chest pain type')
    trestbps = st.text_input('Resting blood pressure')
    chol = st.text_input('Serum cholesterol')
    fbs = st.text_input('Fasting blood sugar')
    restecg = st.text_input('Resting ECG results')
    thalach = st.text_input('Maximum heart rate achieved') 
    exang = st.text_input('Exercise-induced angina')
    oldpeak = st.text_input('ST depression induced by exercise (ECG)')
    slope = st.text_input('Slope of peak exercise ST segment')
    ca = st.text_input('Number of major vessels (0–3) colored by fluoroscopy')
    thal = st.text_input('Thalassemia type')
    
    
    # Prediction code
    
    diagnosis = '' # empty string to store the final outcome
    
    if st.button('Heart Disease Test Result'):
        user_input = [age, sex, cp, trestbps, chol, fbs, restecg,
                  thalach, exang, oldpeak, slope, ca, thal]

        try:
        # Convert all inputs to float explicitly
            numeric_input = [float(val) for val in user_input]

        # Reshape and scale
            np_input = np.asarray(numeric_input).reshape(1, -1)
            scaled_input = loaded_scaler.transform(np_input)

        # Predict
            prediction = loaded_model.predict(scaled_input)

            if prediction[0] == 0:
                diagnosis = '✅ The Person does NOT have a Heart Disease'
                st.success(diagnosis)

            # Tips for staying healthy
                st.markdown("### ✅ Health Tips to Stay Safe:")
                st.markdown("""
                            - 🥗 Maintain a balanced diet (low in salt and saturated fat)  
                            - 🏃 Exercise at least 30 mins daily  
                            - 🚭 Avoid smoking or tobacco  
                            - 🧘 Reduce stress through meditation or hobbies  
                            - 💧 Stay hydrated and get enough sleep  
                            - 📋 Do regular health check-ups
                            """)
            
            if prediction[0] == 1:
                diagnosis = '⚠ The Person HAS Heart Disease'
                st.warning(diagnosis)
    
                # Tips for people at risk
                st.markdown("### ⚠ If You're at Risk of Heart Disease:")
                st.markdown("""
                            - 🩺 *Immediately consult a cardiologist*  
                            - 💊 Follow prescribed medications strictly  
                            - 🍽 Control cholesterol, BP, and blood sugar levels  
                            - 🏃‍♂ Begin heart-friendly physical activity (as advised by doctor)  
                            - 🚫 Quit smoking and alcohol  
                            - 🍵 Follow heart-healthy diet (DASH/Mediterranean diet)
                            """)

        except ValueError:
                diagnosis = "❌ Please enter only valid numbers in all fields."
                st.error(diagnosis)
    

if __name__ == '__main__':
    main()
