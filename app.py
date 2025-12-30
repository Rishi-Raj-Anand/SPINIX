import streamlit as st
from transformers import pipeline

st.set_page_config(layout="wide", page_title="SPINIX")

# Cache model
@st.cache_resource
def load_qa_model():
    return pipeline(
        "question-answering",
        model="distilbert-base-cased-distilled-squad" # default to 30522 tokens
    )

# loading model and caching
qa_pipeline = load_qa_model()

#  initializating session states
if "submitted_context" not in st.session_state:
    st.session_state.submitted_context = ""

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


# Title
st.markdown("<h1 style='text-align: center;'>SPINIX</h1>", unsafe_allow_html=True)

# 2 columns
col_left, col_right = st.columns([1, 1], gap="small")


# left- context part
with col_left:
    st.markdown("<h3 style='text-align: center;'>Context</h3>", unsafe_allow_html=True)
   
    current_context_input = st.text_area(
        "Paste your text below:", 
        height=400, 
        placeholder="Paste your article, document, or text here...",
        label_visibility="collapsed",   
        key="context_input_area"
    )

    # Submit Button
    if st.button("Submit Context", width="stretch"):
        if current_context_input.strip():
            st.session_state.submitted_context = current_context_input
            
            # clear context on submission of new context
            st.session_state.chat_history = [] 
            st.toast("Context submitted successfully! You can now ask questions.")
        else:
            st.toast("Please enter some text first")



#  right - chat
with col_right:
    st.markdown("<h3 style='text-align: center;'>Ask Questions</h3>", unsafe_allow_html=True)

    chat_container = st.container(height=400, border=True)

    # Display Chat History
    with chat_container:
        
        for chat in st.session_state.chat_history:
            with st.chat_message("user"):
                st.write(chat['question'])

            with st.chat_message("assistant"):
                st.write(chat['answer'])

    prompt = st.chat_input("Enter your question...", key="chat_input_widget")

    if prompt:
        
        if not st.session_state.submitted_context:
             st.toast("Please submit a context first")
        else:
            
            with chat_container:
                 with st.chat_message("user"):
                     st.write(prompt)
                 
                 # Process answer
                 with st.chat_message("assistant"):
                     with st.spinner("Thinking..."):
                         # Run model using session state context
                         result = qa_pipeline(
                             question=prompt, 
                             context=st.session_state.submitted_context
                         )
                         
                         answer_text = ""

                         # threshold
                         if result['score'] > 0.1:
                             answer_text = result['answer']
                         else:
                             answer_text = "I cannot find the answer to that question in the provided context."
                         
                         st.write(answer_text)

            # add new question and answer
            st.session_state.chat_history.append(
                {'question': prompt, 'answer': answer_text}
            )

            # Rerun to update new messages
            st.rerun()
