import streamlit as st

if "customer" not in st.session_state:
    st.session_state.customer = False

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

with st.sidebar:
    st.title('🤖🎥 Marketing AI Assistant')
    customer = st.selectbox("Select the customer:",["Meghan","Christina","Avinash"])
    if st.button("Select"):
        st.success('Customer: '+customer+' selected!', icon='✅')
        st.session_state.customer = customer

if prompt := st.chat_input("What is up?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        if str(prompt).lower().count("video")>=1:
            output = "Here's the Hyper-Personalized video for "+st.session_state.customer+":"
        elif str(prompt).lower().count("notice")>=1:
            with open(f"Data/Renewal/{st.session_state.customer}.txt","r",encoding="utf-8") as file:
                output = file.read().replace("\n","\n")
        for response in output:
            full_response += response
            message_placeholder.markdown(full_response + "▌")
        message_placeholder.markdown(output.replace("$","\$").replace('\n', '<br>'),unsafe_allow_html=True)
        if "video" in output:
            st.video("Data/Renewal/output3.mp4")
        st.session_state.response = response
    st.session_state.messages.append({"role": "assistant", "content":output })
