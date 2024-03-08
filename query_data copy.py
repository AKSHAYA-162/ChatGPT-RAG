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

    # To Prepare the database
    embedding_function = OpenAIEmbeddings()
    database = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    # To search the Database
    result = database.similarity_search_with_relevance_scores(query_text, k=3)
    if len(result) == 0 or result[0][1] < 0.7:
        print(f"I'm unable to find the answer to your Question, try another Question.")
        return

    #Template of the content
    context_text = "\n\n\n---\n\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)
    print(prompt)

    #creating a model for open AI
    model = ChatOpenAI()
    response_text = model.predict(prompt)

    #the document source
    sources = [doc.metadata.get("source", None) for doc, _score in results]
    formatted_response = {"response": response_text, "sources": sources}
    # print(formatted_response)
    return formatted_response



#from the "https://docs.streamlit.io/knowledge-base/tutorials/build-conversational-apps#build-a-chatgpt-like-app" website
#
#
st.title("ChatGPT-like clone")

# Set OpenAI API key from Streamlit secrets(toml file)
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# Setting a default model - gpt-3.5-turbo
if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = "gpt-3.5-turbo"

# For initilization chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on the app
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# This step helps accepting the user input
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


