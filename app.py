import os
import time
import streamlit as st
from dotenv import load_dotenv
from serpapi import GoogleSearch
from pymongo import MongoClient
from gridfs import GridFS
from io import BytesIO
from PIL import Image
import requests
from langchain.document_loaders import PDFLoader
from langchain.vectorstores import MongoDBVectorStore
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import OpenAI

# Load environment variables
load_dotenv()

# Streamlit configuration
st.set_page_config(page_title="TroubleShooter", page_icon="images/troubleshootericon.jpeg")
st.title("TroubleShooter")

# MongoDB configuration
MONGO_URI = os.getenv("MONGO_URI")
client = MongoClient(MONGO_URI)
db = client["troubleshooter"]
manuals_fs = GridFS(db)

# SerpApi configuration
SERP_API_KEY = os.getenv("SERP_API_KEY")

# OpenAI configuration
openai_api_key = os.getenv("OPENAI_API_KEY")
llm = OpenAI(api_key=openai_api_key)
embeddings = OpenAIEmbeddings(api_key=openai_api_key)
vector_store = MongoDBVectorStore(client, "troubleshooter", embeddings)

# Streamlit session state initialization
if "messages" not in st.session_state:
    st.session_state.messages = []
if "step" not in st.session_state:
    st.session_state.step = 1
if "device_info" not in st.session_state:
    st.session_state.device_info = {"name": "", "model": ""}

def reset_state():
    st.session_state.messages = []
    st.session_state.step = 1
    st.session_state.device_info = {"name": "", "model": ""}

if st.sidebar.button("New Issue"):
    reset_state()

if st.session_state.device_info["name"] and st.session_state.device_info["model"]:
    st.sidebar.header(f"Device: {st.session_state.device_info['name']} - Model: {st.session_state.device_info['model']}")
    manual_file = manuals_fs.find_one({"filename": f"{st.session_state.device_info['model']}.pdf"})
    if manual_file:
        st.sidebar.download_button(
            label="Download User Manual",
            data=manual_file.read(),
            file_name=f"{st.session_state.device_info['model']}.pdf",
            mime="application/pdf"
        )

for message in st.session_state.messages:
    if message["role"] == "assistant":
        if "image" in message:
            st.chat_message("assistant", avatar=message["avatar"]).write(message["content"])
            st.image(message["image"], caption=message["image_caption"], use_column_width=True)
        else:
            st.chat_message("assistant", avatar=message["avatar"]).write(message["content"])
    else:
        st.chat_message("user", avatar=message["avatar"]).write(message["content"])

def search_manual(model):
    search = GoogleSearch({
        "q": f"{model} user manual",
        "api_key": SERP_API_KEY
    })
    results = search.get_dict()
    return results

def reverse_image_search(image_path):
    search = GoogleSearch({
        "engine": "google_reverse_image",
        "image_url": image_path,
        "api_key": SERP_API_KEY
    })
    results = search.get_dict()
    return results

def generate_response(prompt):
    response = llm(prompt)
    return response["choices"][0]["text"]

def vector_search(query):
    docs = vector_store.similarity_search(query, top_k=5)
    return docs

def generate_steps(docs):
    combined_text = " ".join([doc.page_content for doc in docs])
    prompt = f"Based on the following information, generate a step-by-step troubleshooting guide: {combined_text}"
    response = llm(prompt)
    steps = response["choices"][0]["text"]
    return steps.split("\n")

if st.session_state.step == 1:
    assistant_message = {"role": "assistant", "avatar": "images/robot.png", "content": "What seems to be the problem?"}
    if assistant_message not in st.session_state.messages:
        st.session_state.messages.append(assistant_message)
        st.chat_message("assistant", avatar=assistant_message["avatar"]).write(assistant_message["content"])

# Always display the chat input
user_query = st.chat_input(placeholder="Ask me anything!", key="chat_input")

if user_query:
    user_message = {"role": "user", "avatar": "images/greencircle.jpeg", "content": user_query}
    st.session_state.messages.append(user_message)
    st.chat_message("user", avatar=user_message["avatar"]).write(user_message["content"])

    if st.session_state.step == 1:
        st.session_state.step = 2

    if st.session_state.step == 2:
        if "model" in user_query.lower():
            st.session_state.device_info["model"] = user_query.split()[-1]
            st.session_state.step = 3
        else:
            docs = vector_search(user_query)
            steps = generate_steps(docs)
            st.session_state.steps = steps
            st.session_state.step = 3

    if st.session_state.step == 3:
        current_step = st.session_state.steps[0]
        assistant_message = {"role": "assistant", "avatar": "images/robot.png", "content": current_step}
        if assistant_message not in st.session_state.messages:
            st.session_state.messages.append(assistant_message)
            st.chat_message("assistant", avatar=assistant_message["avatar"]).write(assistant_message["content"])
        st.session_state.steps = st.session_state.steps[1:]
        if not st.session_state.steps:
            st.session_state.step = 99

if st.session_state.step == 99:
    assistant_message = {
        "role": "assistant", 
        "avatar": "images/robot.png", 
        "content": f"I’ve identified your appliance as {st.session_state.device_info['model']}; does this look correct?",
        "image": "images/appliance.png",
        "image_caption": st.session_state.device_info['model']
    }
    if assistant_message not in st.session_state.messages:
        st.session_state.messages.append(assistant_message)
        st.chat_message("assistant", avatar=assistant_message["avatar"]).write(assistant_message["content"])
        st.image(assistant_message["image"], caption=assistant_message["image_caption"], use_column_width=True)
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Yes", key="confirm_model_yes"):
            st.session_state.device_info = {"name": "Appliance", "model": st.session_state.device_info['model']}
            st.session_state.step = 4
    with col2:
        if st.button("No", key="confirm_model_no"):
            assistant_message = {"role": "assistant", "avatar": "images/robot.png", "content": "Sorry about that. Could you provide more details or try uploading another picture?"}
            st.session_state.messages.append(assistant_message)
            st.chat_message("assistant", avatar=assistant_message["avatar"]).write(assistant_message["content"])

if st.session_state.step == 4:
    with st.spinner("Searching for user manual..."):
        time.sleep(5)
    results = search_manual(st.session_state.device_info['model'])
    if "organic_results" in results and len(results["organic_results"]) > 0:
        manual_url = results["organic_results"][0]["link"]
        manual_file = requests.get(manual_url)
        manuals_fs.put(manual_file.content, filename=f"{st.session_state.device_info['model']}.pdf")
        st.session_state.step = 5
    else:
        assistant_message = {"role": "assistant", "avatar": "images/robot.png", "content": "Sorry, I couldn't find the user manual. Please try again later."}
        st.session_state.messages.append(assistant_message)
        st.chat_message("assistant", avatar=assistant_message["avatar"]).write(assistant_message["content"])
        st.session_state.step = 1

if st.session_state.step == 5:
    with st.spinner("Downloading user manual..."):
        time.sleep(3)
    st.session_state.step = 6

if st.session_state.step == 6:
    assistant_message = {"role": "assistant", "avatar": "images/robot.png", "content": "What issue are you having with the device?"}
    if assistant_message not in st.session_state.messages:
        st.session_state.messages.append(assistant_message)
        st.chat_message("assistant", avatar=assistant_message["avatar"]).write(assistant_message["content"])

if st.session_state.step == 7:
    with st.spinner("Searching for solutions..."):
        time.sleep(3)
    assistant_message = {"role": "assistant", "avatar": "images/robot.png", "content": "Is the appliance plugged in?"}
    if assistant_message not in st.session_state.messages:
        st.session_state.messages.append(assistant_message)
        st.chat_message("assistant", avatar=assistant_message["avatar"]).write(assistant_message["content"])

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Yes", key="plugged_in_yes"):
            st.session_state.step = 9
    with col2:
        if st.button("No", key="plugged_in_no"):
            assistant_message = {"role": "assistant", "avatar": "images/robot.png", "content": "Please plug in the appliance and try again."}
            if assistant_message not in st.session_state.messages:
                st.session_state.messages.append(assistant_message)
                st.chat_message("assistant", avatar=assistant_message["avatar"]).write(assistant_message["content"])

if st.session_state.step == 9:
    assistant_message = {"role": "assistant", "avatar": "images/robot.png", "content": "Try resetting the appliance by unplugging it from the outlet, waiting ten seconds, and then plugging it back in. (Refer to the user manual for more details.)"}
    if assistant_message not in st.session_state.messages:
        st.session_state.messages.append(assistant_message)
        st.chat_message("assistant", avatar=assistant_message["avatar"]).write(assistant_message["content"])

    st.write("Are you still experiencing an issue?")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Yes", key="reset_yes"):
            st.session_state.step = 10
    with col2:
        if st.button("No", key="reset_no"):
            st.session_state.step = 10


if st.session_state.step == 10:
    with st.spinner("Searching for solutions..."):
        time.sleep(3)
    assistant_message = {"role": "assistant", "avatar": "images/robot.png", "content": "Verify if the outlet is functioning properly by plugging in another appliance to see if it works. (Refer to the user manual for more details.)"}
    if assistant_message not in st.session_state.messages:
        st.session_state.messages.append(assistant_message)
        st.chat_message("assistant", avatar=assistant_message["avatar"]).write(assistant_message["content"])

    st.write("Are you still experiencing an issue?")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Yes", key="outlet_yes"):
            st.session_state.step = 11
    with col2:
        if st.button("No", key="outlet_no"):
            st.session_state.step = 11

if st.session_state.step == 11:
    with st.spinner("Searching for solutions..."):
        time.sleep(3)
    assistant_message = {"role": "assistant", "avatar": "images/robot.png", "content": "It appears your appliance needs a new fuse."}
    if assistant_message not in st.session_state.messages:
        st.session_state.messages.append(assistant_message)
        st.chat_message("assistant", avatar=assistant_message["avatar"]).write(assistant_message["content"])
    st.session_state.step = 12

if st.session_state.step == 12:
    with st.spinner("Searching for parts..."):
        time.sleep(5)
    assistant_message = {
        "role": "assistant", 
        "avatar": "images/robot.png", 
        "content": "You can purchase a replacement fuse for your appliance at the link below:\n[Replacement Fuse](https://www.example.com)"
    }
    if assistant_message not in st.session_state.messages:
        st.session_state.messages.append(assistant_message)
        st.chat_message("assistant", avatar=assistant_message["avatar"]).write(assistant_message["content"])
