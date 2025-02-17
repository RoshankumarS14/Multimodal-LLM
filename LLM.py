from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from typing import Dict, List


def parse_docs(retriever_output) -> Dict[str, List]:    

    # Initialize lists to store sorted summaries and raw data
    sorted_texts = []
    sorted_tables = []
    sorted_images = []

    # Iterate through the retriever output
    for doc in retriever_output:

            # Organize the data based on the type
            if str(doc)[0] == "/":
                sorted_images.append(doc)
            elif str(doc)[0] == "{":
                sorted_tables.append(doc)
            else:
                sorted_texts.append(str(doc))

    # Return the organized data
    return {
        "texts": sorted_texts,
        "tables": sorted_tables,
        "images": sorted_images,
    }

def build_prompt(kwargs,video_summary):

    docs_by_type = kwargs["context"]
    user_question = kwargs["question"]
    
    # Construct context for text summaries
    context_text = ""
    if len(docs_by_type["texts"]) > 0:
        context_text = "\n".join(docs_by_type["texts"])

    # Construct context for table summaries
    context_tables = ""
    if len(docs_by_type["tables"]) > 0:
        context_tables = "\n".join(docs_by_type["tables"])

    # Construct prompt with context (text, tables, and image summaries)
    prompt_template = f"""
    Answer the question based only on the following context, which can include text, tables, and image summaries.
    Based on analyzing the provided context, answer the question.

    Text Context: {context_text}
    Table Context: {context_tables}
    Video Context" {video_summary}
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