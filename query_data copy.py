import argparse
import os
from dataclasses import dataclass

from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.vectorstores.chroma import Chroma

import streamlit as st
from openai import OpenAI

CHROMA_PATH = "chroma"

PROMPT_TEMPLATE = """
Answer the question based on the following context, if there is no answer based on the context, use your own response:

{context}

---

Answer the question based on the above context: {question}
"""


def main(query_text):
    os.environ["OPENAI_API_KEY"] = "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"

    # Prepare the DB.
    embedding_function = OpenAIEmbeddings()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    # Search the DB.
    results = db.similarity_search_with_relevance_scores(query_text, k=3)
    if len(results) == 0 or results[0][1] < 0.7:
        print(f"Unable to find matching results.")
        return

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)
    print(prompt)

    model = ChatOpenAI()
    response_text = model.predict(prompt)

    sources = [doc.metadata.get("source", None) for doc, _score in results]
    formatted_response = {"response": response_text, "sources": sources}
    # print(formatted_response)
    return formatted_response




st.title("ChatGPT-like clone")

# Set OpenAI API key from Streamlit secrets
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# Set a default model
if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = "gpt-3.5-turbo"

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("What is up?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    out_response = main(prompt)
    print(main(prompt))
    if out_response is None:
        out_response = {}
        out_response["response"] = "Could not find the answer"

        with st.chat_message("assistant"):
            stream = client.chat.completions.create(
                model=st.session_state["openai_model"],
                messages=[
                    {"role": m["role"], "content": m["content"]}
                    for m in st.session_state.messages
                ],
                stream=True,
            )
            response = st.write_stream(stream)
            st.session_state.messages.append({"role": "assistant", "content": response})
    else:
        st.session_state.messages.append(
            {"role": "assistant", "content": out_response["response"]}
        )

        with st.chat_message("assistant"):
            st.markdown(out_response["response"])


