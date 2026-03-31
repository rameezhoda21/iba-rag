import streamlit as st
import sys
from pathlib import Path
import os
from dotenv import load_dotenv

# Add project root to python path to access the app module via imports
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))
load_dotenv(PROJECT_ROOT / ".env")

from app.main import ChatPipeline, startup_event

# Initialize the pipeline state using Streamlit's cache so it doesn't reload heavily every interaction
@st.cache_resource
def load_rag_pipeline() -> ChatPipeline:
    # Set up basic logging if needed or skip startup_event if you just want to init ChatPipeline directly
    import logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    
    # Initialize the ChatPipeline directly since app.main uses a global var pattern for FastAPI
    pipeline = ChatPipeline()
    return pipeline

st.set_page_config(page_title="IBA Student Support Chatbot", page_icon="🎓", layout="centered")

st.title("🎓 IBA Student Support Assistant")
st.markdown("Ask me questions about IBA's fee structures, admission criteria, academic policies, attendance rules, and hostels!")

chatbot_system = load_rag_pipeline()

# Initialize chat history inside Streamlit session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("E.g., What are the hostel charges for BS?"):
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)
    
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Show a loading spinner while the RAG logic processes the request locally
    with st.spinner("Searching IBA handbooks & policies..."):
        try:
            # Query the already-initialized QA system
            response = chatbot_system.ask(query=prompt)
            answer_text = response.answer
            
            # Formulate the response with source tags
            if response.sources:
                sources_str = "\n\n**Sources:**\n" + "\n".join([f"- {s}" for s in response.sources])
                answer_text += sources_str

        except Exception as e:
            answer_text = f"An error occurred while generating reasoning: {e}"

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        st.markdown(answer_text)
        
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": answer_text})