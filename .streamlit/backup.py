# Code refactored from https://docs.streamlit.io/knowledge-base/tutorials/build-conversational-apps

import openai
import streamlit as st
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

with st.sidebar:
    st.title('🤖💬 OpenAI Chatbot')
    if 'OPENAI_API_KEY' in st.secrets:
        st.success('API key already provided!', icon='✅')
        openai.api_key = st.secrets['OPENAI_API_KEY']
    else:
        openai.api_key = st.text_input('Enter OpenAI API token:', type='password')
        if not (openai.api_key.startswith('sk-') and len(openai.api_key)==51):
            st.warning('Please enter your credentials!', icon='⚠️')
        else:
            st.success('Proceed to entering your prompt message!', icon='👉')
    st.selectbox("Select the accident video to analyze:",["Video 1","Video 2"])

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

if "images" not in st.session_state:
    # Extract and encode frames
    frames = extract_frames("Data/videoplayback.mp4", num_frames=20)
    image_elements = [ImageElement(encode_image(frame)) for frame in frames]
    composite_chunk = CompositeElement(image_elements)

    # Now, applying their function
    st.session_state.images = get_images_base64([composite_chunk])
    print("images ready")

if "image_summaries" not in st.session_state:
    st.session_state.image_summaries = get_image_summary(st.session_state.images)
    print("image summaries ready")

if "retriever" not in st.session_state or st.session_state.retriever is None:
    # st.session_state.retriever = get_embedding(st.session_state.images, st.session_state.image_summaries)
    # Load vectorstore from disk
    vectorstore = Chroma(
        collection_name="multi_modal_rag",
        embedding_function=OpenAIEmbeddings(),
        persist_directory="./chroma_db1"
    )

    # Load stored docstore and data
    with open("retriever_data.pkl", "rb") as f:
        data = pickle.load(f)

    store = data["docstore"]
    img_ids = data["img_ids"]
    image_summaries = data["image_summaries"]

    # Recreate the retriever
    st.session_state.retriever = MultiVectorRetriever(vectorstore=vectorstore, docstore=store, id_key="doc_id")
    print("retriever ready")

if "chain" not in st.session_state:
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

if "chain_with_resources" not in st.session_state:
    st.session_state.chain_with_resources = {
        "context": st.session_state.retriever | RunnableLambda(parse_docs),
        "question": RunnablePassthrough(),
    } | RunnablePassthrough().assign(
        response=(
            RunnableLambda(build_prompt)
            | ChatOpenAI(model="gpt-4o-mini")
            | StrOutputParser()
        )
    )
    print("chain with resources ready")

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
        output = st.session_state.chain_with_resources.invoke(prompt)
        st.session_state.button_pressed = False
        for response in output["response"]:
            full_response += response
            message_placeholder.markdown(full_response + "▌")
        message_placeholder.markdown(full_response)
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