import streamlit as st
import joblib
import pandas as pd
from sklearn.preprocessing import LabelEncoder , MinMaxScaler
import requests
from io import BytesIO
import threading
import time
import psutil
import matplotlib.pyplot as plt

def simulate_edge_resource_limit():
    def throttled_cpu():
        while True:
            start = time.time()
            while time.time() - start < 0.3:  # aktif 30%
                pass
            time.sleep(0.7)  # idle 70%

    def monitor():
        while True:
            print(f"[CPU]: {psutil.cpu_percent()}% | [RAM]: {psutil.virtual_memory().percent}%")
            time.sleep(1)

    # Run di background
    threading.Thread(target=throttled_cpu, daemon=True).start()
    threading.Thread(target=monitor, daemon=True).start()


@st.cache_resource
def load_model_and_encoders():
    model = joblib.load('Personality-prediction-model.pkl')
    encoders = joblib.load('encoder.pkl')
    return model, encoders

model, encoders = load_model_and_encoders()

if "execution_history" not in st.session_state:
    st.session_state.execution_history = []

def convert_to_df(time_spent_alone , stage_fear , social_event_attendance , going_outside , drained_after_socializing , friend_circle_size , post_frequency):
    df = pd.DataFrame({
    "Time_spent_Alone" : [time_spent_alone] , 
    "Stage_fear" : [stage_fear] , 
    "Social_event_attendance" : [social_event_attendance] , 
    "Going_outside" : [going_outside] , 
    "Drained_after_socializing" : [drained_after_socializing] , 
    "Friends_circle_size" : [friend_circle_size] , 
    "Post_frequency" : [post_frequency],
    })
    
    return df

def predict(time_spent_alone , stage_fear , social_event_attendance , going_outside , drained_after_socializing , friend_circle_size , post_frequency):
    df = convert_to_df(time_spent_alone , stage_fear , social_event_attendance , going_outside , drained_after_socializing , friend_circle_size , post_frequency)
    
    for col, enc in encoders.items():
      if col in df.columns :
        if isinstance(enc , LabelEncoder) :
          df[col] = enc.transform(df[col])
          
        else:
          df[col] = enc.transform(df[[col]])
 
            
    
    pred = model.predict(df)
    pred = encoders['Personality'].inverse_transform(pred)
    st.success(f"Personality : {pred[0]}")

def predict_batch_locally(csv_file):
    contents = csv_file.read()
    df = pd.read_csv(BytesIO(contents))
    result_df  = df.copy()
    
    for col, enc in encoders.items():
      if col in df.columns :
        if isinstance(enc , LabelEncoder) :
          df[col] = enc.transform(df[col])
          
        else:
          df[col] = enc.transform(df[[col]])
 
            
    
    pred = model.predict(df)
    pred = encoders['Personality'].inverse_transform(pred)
    result_df['Personality'] = pred
    
    return result_df.to_dict(orient="records")
            


st.title("Personality Prediction")
st.markdown("### Via CSV file")
    
uploaded_file = st.file_uploader("Choose a file" , type = ['csv' , 'xls' , 'tsv'])
    
if uploaded_file is not None:
    # Can be used wherever a "file-like" object is accepted:
    dataframe = pd.read_csv(uploaded_file)
    st.write(dataframe.head())
    predict_button = st.button("Predict")
    force_inference_in_edge = st.checkbox("Force Inference process to use edge resources")

    if predict_button and force_inference_in_edge == False :
        start = time.time()
        with st.spinner("Mengirim ke cloud..."):
            try:
                uploaded_file.seek(0)

                response = requests.post(
                    "https://sister-api-922886404287.asia-southeast2.run.app/predict",
                    files={"csv_file": (uploaded_file.name, uploaded_file, "text/csv")},
                )
                st.write(response.status_code)
                if response.status_code == 200:
                    st.success("Berhasil prediksi dari cloud!")
                    response_json = response.json()
                    end = time.time()
                    st.session_state.execution_history .append(("Cloud",end-start , uploaded_file.name))
                    st.write(pd.DataFrame(response_json))


            except Exception as e:
                st.error("Terjadi kesalahan , error " , e )
    elif predict_button and force_inference_in_edge:
        simulate_edge_resource_limit()
        start = time.time()
        with st.spinner("Memproses lokal dengan pembatasan edge..."):
            uploaded_file.seek(0)
            time.sleep(4)
            response = predict_batch_locally(uploaded_file)
            response_json = response
            end = time.time()
            
            st.session_state.execution_history .append(("Edge",end-start , uploaded_file.name))
            st.write(pd.DataFrame(response_json))
        
    
    
if st.session_state.execution_history:
    st.markdown("### 📈 Execution Time Trend")

    df_time = pd.DataFrame(st.session_state.execution_history, columns=["Mode", "Time (s)", "Filename"])
    

    fig, ax = plt.subplots()

    for mode in df_time["Mode"].unique():
        subset = df_time[df_time["Mode"] == mode]
        ax.plot(subset["Filename"], subset["Time (s)"], marker='o', label=mode)

    ax.set_xlabel("Filename")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
    ax.set_ylabel("Time (seconds)")
    ax.set_title("Trend of Inference Time based on file")
    ax.legend()
    ax.grid(True)
  
    st.pyplot(fig)

              

    
with st.form("Value_form"):

    
    
    st.markdown("### Via Manual Input")
    time_spent_alone = st.number_input("Time spent alone", min_value=0)
    stage_fear = st.selectbox("Stage fear" , ("Yes" , "No"))
    social_event_attendance = st.number_input("Social event attendance", min_value=0)
    going_outside = st.number_input("Going outside", min_value=0)
    drained_after_socializing = st.selectbox("Drained after socializing" , ("Yes" , "No"))
    friend_circle_size = st.number_input("Friend circle size", min_value=0)
    post_frequency = st.number_input("Post frequency", min_value=0)
    submitted = st.form_submit_button("Predict")
    
    if submitted : 
        predict(time_spent_alone , stage_fear , social_event_attendance , going_outside , drained_after_socializing , friend_circle_size , post_frequency)
