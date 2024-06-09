import streamlit as st
import json
import dspy
from dspy.teleprompt import BootstrapFewShot
from dspy.evaluate.evaluate import Evaluate


uploaded_file = st.file_uploader("Upload Files",type=['json'])

if uploaded_file is None:
    st.stop()

if uploaded_file is not None:
    data = json.load(uploaded_file)
    st.write(data)


lm = dspy.HFClientVLLM(model="microsoft/Phi-3-medium-128k-instruct", port=38242, url="http://localhost", max_tokens=200)
