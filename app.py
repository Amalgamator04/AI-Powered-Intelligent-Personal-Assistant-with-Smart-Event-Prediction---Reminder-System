# app.py
import streamlit as st
from config import Config
from database.vector_store import VectorStore
from database.session_manager import SessionManager
from agent.personal_agent import PersonalAgent
from helper.speechtotext import voice_search

# ---------- Init ----------
st.set_page_config(page_title="Personal AI Agent", layout="wide")

# Add a hidden button to clear cache if needed
if 'cache_cleared' not in st.session_state:
    st.session_state.cache_cleared = False

@st.cache_resource
def init_agent(model):
    Config.create_dirs()
    vector_store = VectorStore(str(Config.VECTOR_DB_PATH))
    session_manager = SessionManager(str(Config.METADATA_DB_PATH))
    return PersonalAgent(vector_store, session_manager, llm_model=model)

# ---------- Sidebar ----------
st.sidebar.title("⚙️ Settings")

mode = st.sidebar.selectbox(
    "Mode",
    ["add", "query", "chat", "stats"]
)

input_type = st.sidebar.selectbox(
    "Input Type",
    ["text", "voice"]
)

model = st.sidebar.text_input(
    "Model",
    value=Config.LLM_MODEL
)

temperature = st.sidebar.slider(
    "Temperature",
    0.0, 1.0, 0.7
)

# ---------- Database Reset Section ----------
st.sidebar.markdown("---")
st.sidebar.subheader("🗑️ Database Management")

col1, col2 = st.sidebar.columns(2)

with col1:
    if st.button("Reset All Data", key="reset_all_button"):
        with st.spinner("Resetting database..."):
            try:
                # Create a temporary agent to perform reset
                reset_agent = init_agent(model)
                result = reset_agent.reset_all()
                st.sidebar.success("✅ Database reset successfully!")
                st.sidebar.info(
                    f"Deleted:\n"
                    f"- {result['vector_documents_deleted']} knowledge base documents\n"
                    f"- {result['sessions_deleted']} sessions\n"
                    f"- {result['messages_deleted']} messages"
                )
                st.rerun()
            except Exception as e:
                st.sidebar.error(f"❌ Error during reset: {str(e)}")

with col2:
    if st.button("🔄 Clear Cache", key="clear_cache_button"):
        st.cache_resource.clear()
        st.sidebar.success("✅ Cache cleared!")
        st.sidebar.info("Please refresh the page to reload the agent.")


st.sidebar.markdown("---")

agent = init_agent(model)

st.title("🧠 Personal AI Knowledge Base")

# ---------- Stats ----------
if mode == "stats":
    st.subheader("📊 Knowledge Base Stats")
    stats = agent.get_stats()
    st.json(stats)

# ---------- Add ----------
elif mode == "add":
    st.subheader("➕ Add to Knowledge Base")

    source = st.text_input("Source", value="manual")
    uploaded_file = st.file_uploader("Upload text file", type=["txt"])
    text_input = st.text_area("Enter text", height=200)

    if st.button("Add"):
        text = None

        if uploaded_file:
            text = uploaded_file.read().decode("utf-8")
        elif input_type == "voice":
            with st.spinner("🎤 Listening..."):
                text = voice_search()
        else:
            text = text_input

        if not text:
            st.error("No input provided")
        else:
            with st.spinner("Adding to knowledge base..."):
                doc_ids = agent.add_to_knowledge_base(
                    text,
                    source=source,
                    metadata={
                        "input_type": input_type,
                        "file": uploaded_file.name if uploaded_file else None
                    }
                )
            st.success(f"Added {len(doc_ids)} chunks")

# ---------- Query ----------
elif mode == "query":
    st.subheader("🔍 Query Knowledge Base")

    question = st.text_input("Your question")

    if st.button("Search"):
        if input_type == "voice":
            with st.spinner("🎤 Listening..."):
                question = voice_search()

        if not question:
            st.error("No question provided")
        else:
            with st.spinner("Searching..."):
                result = agent.query(question)

            st.markdown(f"### ❓ {result['question']}")
            for i, (chunk, dist) in enumerate(
                zip(result["context"], result["distances"]), 1
            ):
                st.markdown(f"**[{i}] Similarity:** `{1 - dist:.3f}`")
                st.code(chunk[:500])

# ---------- Chat ----------
elif mode == "chat":
    st.subheader("💬 Chat")

    if "session_id" not in st.session_state:
        st.session_state.session_id = agent.start_session(
            {"mode": "chat", "input_type": input_type}
        )

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_input = st.chat_input("Type your message")

    if user_input or input_type == "voice":
        if input_type == "voice" and not user_input:
            with st.spinner("🎤 Listening..."):
                user_input = voice_search()

        if user_input:
            st.session_state.messages.append(
                {"role": "user", "content": user_input}
            )

            with st.chat_message("user"):
                st.markdown(user_input)

            with st.spinner("Thinking..."):
                response = agent.chat(user_input)

            st.session_state.messages.append(
                {"role": "assistant", "content": response}
            )

            with st.chat_message("assistant"):
                st.markdown(response)
