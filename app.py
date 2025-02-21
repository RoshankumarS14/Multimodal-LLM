# Code refactored from https://docs.streamlit.io/knowledge-base/tutorials/build-conversational-apps

import openai
import streamlit as st
import os
import base64
from io import BytesIO
from PIL import Image
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain.retrievers import MultiVectorRetriever
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
import pickle
from Video_Processing import extract_frames, encode_image, get_images_base64
from Embedding import get_image_summary, get_embedding
from LLM import parse_docs, build_prompt


if "pickles" not in st.session_state:
    # Define the directory
    folder_path = 'Vector Database'

    # Get all .pkl files in the folder
    st.session_state.pickles = [f[:-4] for f in os.listdir(folder_path) if f.endswith('.pkl')]

def build_prompt_with_param(custom_param):
    return RunnableLambda(lambda data: build_prompt(data, custom_param))

def load_retriever_and_chain(case):
    print("Button clicked")
    print("Pickle:",case)
    # Load vectorstore from disk
    vectorstore = Chroma(
        collection_name="multi_modal_rag",
        embedding_function=OpenAIEmbeddings(),
        persist_directory=f"./Vector Database/{case[:-4]}"
    )

    # Load stored docstore and data
    with open(f"./Vector Database/{case}", "rb") as f:
        data = pickle.load(f)

    store = data["docstore"]
    st.session_state.video_summary = data["video_summary"]

    # Recreate the retriever
    st.session_state.retriever = MultiVectorRetriever(vectorstore=vectorstore, docstore=store, id_key="id")
    print("retriever ready")

# if "chain" not in st.session_state:
    st.session_state.chain = (
        {
            "context": st.session_state.retriever | RunnableLambda(parse_docs),
            "question": RunnablePassthrough(),
        }
        | RunnableLambda(build_prompt)
        | ChatOpenAI(model="gpt-4o-mini")
        | StrOutputParser()
    )
    print("chain ready")

# if "chain_with_resources" not in st.session_state:
    st.session_state.chain_with_resources = {
        "context": st.session_state.retriever | RunnableLambda(parse_docs),
        "question": RunnablePassthrough(),
    } | RunnablePassthrough().assign(
        response=(
            build_prompt_with_param(st.session_state.video_summary)
            | ChatOpenAI(model="gpt-4o-mini")
            | StrOutputParser()
        )
    )
    print("chain with resources ready")

with st.sidebar:
    st.title('🤖💬 OpenAI Chatbot')
    openai.api_key = os.getenv("OPENAI_API_KEY")
    st.success('API key already provided!', icon='✅')
     
    case = st.selectbox("Select the case to analyze:",st.session_state.pickles)
    if st.button("load"):
        load_retriever_and_chain(case+".pkl")

if "messages" not in st.session_state:
    st.session_state.messages = []

class ImageElement:
    def __init__(self, image_b64):
        self.metadata = ImageMetadata(image_b64)

class ImageMetadata:
    def __init__(self, image_b64):
        self.image_base64 = image_b64

class CompositeElement:
    def __init__(self, elements):
        self.metadata = CompositeMetadata(elements)

class CompositeMetadata:
    def __init__(self, elements):
        self.orig_elements = elements

if "button_pressed" not in st.session_state:
    st.session_state.button_pressed=False

if "response_returned" not in st.session_state:
    st.session_state.response_returned=False

if "response" not in st.session_state:
    st.session_state.response=""

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("What is up?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        print("Prompt entered:",prompt)
        # output = st.session_state.chain_with_resources.invoke(prompt)
        output = st.session_state.chain_with_resources.invoke(prompt)
        st.session_state.button_pressed = False
        for response in output["response"]:
            full_response += response
            message_placeholder.markdown(full_response + "▌")
        message_placeholder.markdown(full_response.replace("$","\$"))
        st.session_state.response = output
        st.session_state.response_returned = True
        # if st.button("Get clips"):
        #     st.text("Button pressed")
        #     st.session_state.button_pressed = True
    st.session_state.messages.append({"role": "assistant", "content": full_response})

if st.session_state.response_returned:
    if st.button("Get clips"):
        st.session_state.button_pressed = True

if st.session_state.button_pressed:
    for image in st.session_state.response['context']['images']:
        image_data = base64.b64decode(image)
        # Convert to a PIL image
        image = Image.open(BytesIO(image_data))
        # Display in Streamlit
        st.image(image, caption="Decoded Image", use_column_width=True)
