from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
import base64
from base64 import b64decode


# def parse_docs(docs):
#     """Split base64-encoded images and texts"""
#     b64 = []
#     text = []
#     for doc in docs:
#         try:
#             b64decode(doc)
#             b64.append(doc)
#         except Exception as e:
#             text.append(doc)
#     return {"images": b64, "texts": text}

# import base64

def parse_docs(docs):
    """Splits base64-encoded images and text, and sorts text summaries by frame_id."""
    b64 = []
    text_with_metadata = []

    for doc in docs:
        try:
            # Attempt to decode Base64 to check if it's an image
            base64.b64decode(doc)
            b64.append(doc)  # Store as an image
        except Exception:
            # If not Base64, treat as text and store metadata for sorting
            text_with_metadata.append((doc.metadata.get("frame_id", float("inf")), doc))

    # Sort text summaries by frame_id to preserve order
    text_with_metadata.sort(key=lambda x: x[0])  # Sort based on frame_id

    # Extract only sorted text contents
    sorted_texts = [content for _, content in text_with_metadata]

    return {"images": b64, "texts": sorted_texts}

def build_prompt(kwargs):

    docs_by_type = kwargs["context"]
    user_question = kwargs["question"]

    context_text = ""
    if len(docs_by_type["texts"]) > 0:
        for text_element in docs_by_type["texts"]:
            context_text += text_element.text

    # construct prompt with context (including images)
    prompt_template = f"""
    Based on analyzing frames of the video, Answer the question
    Context: {context_text}
    Question: {user_question}
    """

    prompt_content = [{"type": "text", "text": prompt_template}]

    if len(docs_by_type["images"]) > 0:
        for image in docs_by_type["images"]:
            prompt_content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{image}"},
                }
            )

    return ChatPromptTemplate.from_messages(
        [
            HumanMessage(content=prompt_content),
        ]
    )