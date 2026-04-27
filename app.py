import time
import streamlit as st
import uuid
from agent import LangGraphAgent

st.set_page_config(page_title="Movie Chatbot", page_icon="🎬", layout="wide")
st.title("Movie Chatbot")
st.write("Ask me anything about movies!")

if "agent" not in st.session_state:
    with st.spinner("Initializing AI Agent..."):
        st.session_state.agent = LangGraphAgent()

with st.sidebar:
    st.header("Session Configuration")
    user_thread_id = st.text_input("Session ID", value="default_session")
    st.write("Use the same Session ID to resume memory/history across runs.")

if "thread_id" not in st.session_state or st.session_state.thread_id != user_thread_id:
    st.session_state.thread_id = user_thread_id
    st.session_state.messages = []

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("What would you like to know about movies?"):
    with st.chat_message("user"):
        st.markdown(prompt)

    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        status_container = st.container()

        input_data = {"messages": [("user", prompt)]}
        config = {"configurable": {"thread_id": st.session_state.thread_id}}

        final_response = "No response generated."

        try:
            for event in st.session_state.agent.app.stream(input_data, config):
                for node_name, node_output in event.items():
                    if "messages" in node_output and node_output["messages"]:
                        last_message = node_output["messages"][-1]

                        if (
                            node_name == "agent"
                            and hasattr(last_message, "tool_calls")
                            and last_message.tool_calls
                        ):
                            with status_container:
                                for tool_call in last_message.tool_calls:
                                    tool_name = tool_call.get("name", "Unknown tool")
                                    st.status(
                                        f"🔧 Using tool: **{tool_name}**...",
                                        state="complete",
                                    )

                        if node_name == "agent" and not (
                            hasattr(last_message, "tool_calls")
                            and last_message.tool_calls
                        ):
                            final_response = last_message.content

            def stream_data():
                for word in str(final_response).split(" "):
                    yield word + " "
                    time.sleep(0.02)

            st.write_stream(stream_data)

            st.session_state.messages.append(
                {"role": "assistant", "content": final_response}
            )

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
