import os

from openai import OpenAI
import streamlit as st

os.environ.setdefault("OPENAI_API_KEY", "sk-cc2df096be3c4382b3da9a63e5b3b267")
os.environ.setdefault("BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")

with st.sidebar:
    openai_api_key = st.text_input("OpenAI API Key", key="chatbot_api_key", type="password")
    openai_api_url = st.text_input("OpenAI API Url", key="chatbot_api_url", type="default")
    openai_model = st.text_input("OpenAI API Model", key="chatbot_api_model", type="default")

    "[Get an OpenAI API key](https://platform.openai.com/account/api-keys)"
    "[View the source code](https://github.com/streamlit/llm-examples/blob/main/Chatbot.py)"
    "[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/streamlit/llm-examples?quickstart=1)"

st.title("💬 Chatbot")
st.caption("🚀 A Streamlit chatbot powered by OpenAI")
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input():
    api_key = openai_api_key or os.environ.get("OPENAI_API_KEY")
    base_url = openai_api_url or os.environ.get("BASE_URL")
    if not api_key:
        st.info("Please add your OpenAI API key to continue.")
        st.stop()

    client = OpenAI(
        api_key=api_key,
        base_url=base_url,
    )

    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)
    response = client.chat.completions.create(
        model=openai_model or "qwen-plus",
        messages=[
            {'role': 'system', 'content': 'You are a helpful assistant.'},
            {'role': 'user', 'content': prompt}],
    )
    msg = response.choices[0].message
    st.session_state.messages.append({"role": "assistant", "content": msg.content or "未知错误"})
    st.chat_message("assistant").write(msg.content or "未知错误")
