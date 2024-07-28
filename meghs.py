import streamlit as st
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import joblib
import re

# Load the data
data = pd.read_csv('output.csv')

# Ensure all features are numeric
X = data.drop(['Diseases', 'Medication_Tablet', 'Medication_Injection', 'Predicted_Disease'], axis=1)
y = data['Diseases']

# Convert all feature columns to numeric if needed
X = X.apply(pd.to_numeric, errors='coerce')
X.fillna(0, inplace=True)  # Handle any remaining non-numeric values by replacing with 0

# Encode the target variable (Disease)
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Initialize and train the Decision Tree Classifier
clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)

# Save the trained model and label encoder
joblib.dump(clf, 'trained_model.pkl')
joblib.dump(label_encoder, 'label_encoder.pkl')

# Create a dictionary for medication lookup
medication_dict = data.set_index('Diseases')[['Medication_Tablet', 'Medication_Injection']].to_dict('index')

# Function to predict disease and provide medication
def predict_disease(symptoms):
    # Create a dataframe from the symptoms dictionary
    input_data = pd.DataFrame([symptoms])
    
    # Ensure all features are numeric
    input_data = input_data.apply(pd.to_numeric, errors='coerce')
    input_data.fillna(0, inplace=True)  # Handle any remaining non-numeric values by replacing with 0

    # Ensure the input data has the same columns as the training data
    missing_cols = set(X.columns) - set(input_data.columns)
    for col in missing_cols:
        input_data[col] = 0
    input_data = input_data[X.columns]  # Reorder columns to match training data
    
    # Load the trained model and label encoder
    model = joblib.load('trained_model.pkl')
    label_encoder = joblib.load('label_encoder.pkl')
    
    # Predict the disease
    predicted_label = model.predict(input_data)[0]
    predicted_disease = label_encoder.inverse_transform([predicted_label])[0]
    
    # Lookup medication
    medication = medication_dict.get(predicted_disease, {'Medication_Tablet': 'Unknown', 'Medication_Injection': 'Unknown'})
    
    return predicted_disease, medication['Medication_Tablet'], medication['Medication_Injection']

# Streamlit app
def main():
    st.title("DIAGNOCURE")

    # Login page
    if 'logged_in' not in st.session_state:
        st.session_state['logged_in'] = False
    
    if not st.session_state['logged_in']:
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        if st.button("Login"):
            # Simple username/password check
            if username == "meghana" and password == "123456":
                st.session_state['logged_in'] = True
                st.success("Logged in successfully")
            else:
                st.error("Invalid username or password")
    else:
        st.write("Welcome to the Disease Prediction System")
        user_input = st.text_area("Please enter your symptoms separated by commas:").lower()
        
        if st.button("Predict"):
            if user_input:
                symptoms = {}
                user_symptoms = [symptom.strip() for symptom in re.split(',|;', user_input)]
                
                for symptom in X.columns:
                    symptoms[symptom] = 1 if symptom.lower() in user_symptoms else 0
                
                disease, tablet, injection = predict_disease(symptoms)
                st.write(f"**Predicted Disease:** {disease}")
                st.write(f"**Recommended Tablet:** {tablet}")
                st.write(f"**Recommended Injection:** {injection}")
            else:
                st.error("Please enter your symptoms")

if __name__ == "__main__":
    main()
