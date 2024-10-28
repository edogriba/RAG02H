import logging
import streamlit as st

from ui.initializer import customize, initialize
from ui.utils import StreamlitLogHandler, create_log_handler

# Initialize session state for themes
ms = st.session_state
if "themes" not in ms: 
    ms.themes = {
        "current_theme": "light",
        "refreshed": True,
        "light": {
            "theme.base": "dark",
            "theme.backgroundColor": "black",
            "theme.primaryColor": "#c98bdb",
            "theme.secondaryBackgroundColor": "#5591f5",
            "theme.textColor": "white",
            "button_face": "🌜"
        },
        "dark": {
            "theme.base": "light",
            "theme.backgroundColor": "white",
            "theme.primaryColor": "#5591f5",
            "theme.secondaryBackgroundColor": "#82E1D7",
            "theme.textColor": "#0a1464",
            "button_face": "🌞"
        }
    }

def ChangeTheme():
    previous_theme = ms.themes["current_theme"]
    tdict = ms.themes["light"] if ms.themes["current_theme"] == "light" else ms.themes["dark"]
    
    # Set Streamlit options for the selected theme
    for vkey, vval in tdict.items(): 
        if vkey.startswith("theme"):
            st._config.set_option(vkey, vval)

    ms.themes["refreshed"] = False
    ms.themes["current_theme"] = "light" if previous_theme == "dark" else "dark"

# Change the button face based on the current theme
btn_face = ms.themes["light"]["button_face"] if ms.themes["current_theme"] == "light" else ms.themes["dark"]["button_face"]

if __name__ == "__main__":
    # Initialize resources
    resources = initialize()
    customize()

    
    # Header Section
    st.markdown(
        "<h1 style='text-align: center;'>RAG Chatbot</h1>",
        unsafe_allow_html=True
    )
    # Displaying the header image (update the path to your image)
    st.image("INTESA-SANPAOLO_COL.png", use_column_width=True)


    # Theme Toggle Button
    st.button(btn_face, on_click=ChangeTheme)

    # Ingestion Section
    st.markdown("<div style='background-color: #333; color: white; padding: 10px;'>Ingestion</div>", unsafe_allow_html=True)

    if "ongoing_ingestion" not in st.session_state:
        st.session_state["ongoing_ingestion"] = False

    def start_ingestion(keyword):
        if keyword:  # Ensure that the keyword is not empty
            st.session_state.ongoing_ingestion = True
            resources.setup_task_logger(
                handlers=[
                    create_log_handler(StreamlitLogHandler, resources.log_formatter, log_container.code),
                    create_log_handler(logging.StreamHandler, resources.log_formatter),
                ]
            )
            resources.ingest(keyword=keyword)
            st.session_state.ongoing_ingestion = False
            resources.setup_task_logger(
                handlers=[
                    create_log_handler(logging.StreamHandler, resources.log_formatter)
                ]
            )
            # Clear previous message history
            if "messages" in st.session_state:
                st.session_state["messages"] = []

            if "user_question" in st.session_state:
                st.session_state.user_question = None
            
            st.experimental_rerun()

    # Text input for the keyword
    keyword_input = st.text_input(
        "Insert your keyword for the ingestion:", 
        "", 
        key="keyword", 
        on_change=lambda: start_ingestion(st.session_state.keyword) if st.session_state.keyword else None
    )

    log_container = st.empty()

    # Button for starting ingestion
    st.button(
        "Start Ingestion",
        on_click=lambda: start_ingestion(st.session_state.keyword),
        disabled=st.session_state.ongoing_ingestion,
    )

    # Retrieval Section
    st.markdown("<div style='background-color: #333; color: white; padding: 10px;'>Retrieval</div>", unsafe_allow_html=True)

    def get_answer(question):
        response = resources.llm_gen_answer(question=question)
        return response

    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    if "user_question" not in st.session_state:
        st.session_state.user_question = None

    def submit_user_question():
        st.session_state.user_question = st.session_state.widget
        st.session_state.widget = ""

    opening_msg = "Hi! If you have already completed the ingestion phase, write your question here, then press Enter. No memory of previous messages is retained."
    st.text_input(opening_msg, "", key="widget", on_change=submit_user_question)

    if st.session_state.user_question:
        answer = get_answer(st.session_state.user_question)

        st.session_state["messages"].append(
            {"question": st.session_state.user_question, "answer": answer}
        )
        st.session_state.user_question = None

    # Display the conversation history
    if st.session_state["messages"]:
        st.write("### Conversation")
        for chat in st.session_state["messages"]:
            st.write(f"**You**: {chat['question']}")
            st.write(f"**Bot**: {chat['answer']}")
            st.write("---")

    if not ms.themes["refreshed"]:
        ms.themes["refreshed"] = True
        st.experimental_rerun()
