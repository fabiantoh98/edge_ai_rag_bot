import streamlit as st
import requests
from datetime import datetime

API_URL = "http://localhost:8000"

# Initialize chat history
if 'qa_history' not in st.session_state:
    st.session_state.qa_history = []

st.title("Edge AI QA Bot")

# Chat interface
st.header("Ask Me anything about Edge AI")

# Display chat history using st.chat_message
if st.session_state.qa_history:
    for qa in st.session_state.qa_history:
        # Display user message
        with st.chat_message("user"):
            st.write(qa['question'])
            st.caption(f"Asked at: {qa['timestamp']}")
        
        # Display assistant response
        with st.chat_message("assistant"):
            st.write(qa['answer'])

# Chat input
if prompt := st.chat_input("Enter your question about Edge AI..."):
    # Add user message to chat history immediately
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Display user message
    with st.chat_message("user"):
        st.write(prompt)
        st.caption(f"Asked at: {timestamp}")
    
    # Get response from backend
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response = requests.post(f"{API_URL}/query", json={"query": prompt})
                if response.status_code == 200:
                    answer = response.json()["answer"]
                    st.write(answer)
                    
                    # Store in session state
                    st.session_state.qa_history.append({
                        "question": prompt,
                        "answer": answer,
                        "timestamp": timestamp
                    })
                else:
                    error_msg = f"Error from backend: {response.text}"
                    st.error(error_msg)
                    
                    # Store error in history too
                    st.session_state.qa_history.append({
                        "question": prompt,
                        "answer": error_msg,
                        "timestamp": timestamp
                    })
            except Exception as e:
                error_msg = f"Connection error: {str(e)}"
                st.error(error_msg)
                
                # Store error in history
                st.session_state.qa_history.append({
                    "question": prompt,
                    "answer": error_msg,
                    "timestamp": timestamp
                })

# Sidebar for additional controls
with st.sidebar:
    st.header("Chat Controls")
    
    # Clear chat history
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.qa_history = []
        st.rerun()
    
    # Display chat stats
    if st.session_state.qa_history:
        st.metric("Total Questions", len(st.session_state.qa_history))
        
        # Export chat history
        if st.button("📥 Export Chat History", use_container_width=True):
            chat_text = ""
            for i, qa in enumerate(st.session_state.qa_history, 1):
                chat_text += f"Q{i} ({qa['timestamp']}): {qa['question']}\n"
                chat_text += f"A{i}: {qa['answer']}\n\n"
            
            st.download_button(
                label="Download as TXT",
                data=chat_text,
                file_name=f"edge_ai_chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain"
            )