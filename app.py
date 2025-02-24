import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper
from langchain_community.tools import WikipediaQueryRun, ArxivQueryRun,DuckDuckGoSearchRun
from langchain.agents import initialize_agent,AgentType
from langchain.callbacks import StreamlitCallbackHandler
import os
from dotenv import load_dotenv

#ArXiv tool
arxiv_wrapper = ArxivAPIWrapper(top_k_results=1,doc_content_chars_max=250)
arxiv = ArxivQueryRun(api_wrapper=arxiv_wrapper)

#Wikipedia tool
wiki_wrapper = WikipediaAPIWrapper(top_k_results=1,doc_content_chars_max=250)
wiki = WikipediaQueryRun(api_wrapper=wiki_wrapper)


search = DuckDuckGoSearchRun(name="Search")

st.image("image.png", width=160)
st.title("Search in Wikipedia & Arxiv")
"""
In this example, we're using `StreamlitCallbackHandler` to display the thought and action of an agent in an interactive Streamlit app.
Try more Lanchain Streamlit Agent examples at [github.com/langchain-ai/streamlit-agent](https://github.com/langchain-ai/streamlit-agent).
"""

#SLIDER

st.sidebar.title("Settings")
api_key=st.sidebar.text_input("Enter your Groq API Key:", type="password")

if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role":"assistant",
         "content":"Hi, I'm chatbot who can search the web. How can I"
         }
    ]
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg['content'])

if prompt:= st.chat_input(placeholder="What is machine learning"):
    st.session_state.messages.append({"role":"user", "content":prompt})
    st.chat_message("user").write(prompt)

    llm=ChatGroq(groq_api_key=api_key, model_name="Llama3-8b-8192", streaming=True)
    tools=[search, arxiv, wiki]

    search_agent= initialize_agent(tools=tools,
                                   llm=llm,
                                   agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                                   handle_parsing_errors=True)

    with st.chat_message("assistant"):
        st_cb=StreamlitCallbackHandler(st.container(),expand_new_thoughts=False)
        response=search_agent.run(st.session_state.messages,callbacks=[st_cb])
        st.session_state.messages.append({'role':'assistant',
                                          "content":response})
        st.write(response)

