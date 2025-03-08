import streamlit as st
from integration.multimodal_integration import process_inputs

st.title("MediVoice+Vision: AI-Powered Symptom Checker")

# Input fields
audio_file = st.file_uploader("Upload Audio File", type=["mp3", "wav"])
image_file = st.file_uploader("Upload Image File", type=["jpg", "png"])

if st.button("Analyze"):
    if audio_file and image_file:
        result = process_inputs(audio_file, image_file)
        st.write("### Symptom Analysis")
        st.json(result["symptoms"])
        st.write("### Image Analysis")
        st.json(result["image_analysis"])
    else:
        st.error("Please upload both audio and image files.")