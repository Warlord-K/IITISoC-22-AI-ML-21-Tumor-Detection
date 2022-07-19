from prediction_page import show_prediction_page
import streamlit as st

st.header("Disease Detection Deep Learning Models")

models = ["Tumor Detection","Heart Detection","Any other Detection"]
models_info = ["Info about tumor Detection","Info about heart detection","Info about other detection"]
press = [False]*len(models)
with st.sidebar:
    st.title("Browse Models")
    for i,model in enumerate(models):
        press[i] = st.sidebar.button(model)
        with st.expander("See Info"):
            st.write(models_info[i])
    
if sum(press) == 0:
    press[0] = True

for pres in press:
    if pres:
        show_prediction_page()