import streamlit as st
from hugchat import hugchat
from hugchat.login import Login

# App title
st.set_page_config(page_title=f"{''':handshake:'''} Chatbot Meli {''':handshake:'''}")

# Hugging Face Credentials
with st.sidebar:
    st.title(f"{''':handshake:'''} Chatbot Meli {''':handshake:'''}")
    if 'hf_email' not in st.session_state or 'hf_pass' not in st.session_state:
        st.write("⚠️ Debes registrarte en HugginFace para usar esta app. Puedes registrarte aquí [🤗](https://huggingface.co/join).")
        hf_email = st.text_input('Ingrese su E-mail:', type='default')
        hf_pass = st.text_input('Ingrese su password:', type='password')
        if not (hf_email and hf_pass):
            st.warning('Por favor ingresa tus credenciales!', icon='⚠️')
        else:
            st.success('Listo! Comienza a conversar con Meli, tu asistente', icon='👉')


            with st.expander("ℹ️ Filtros avanzados"):
                condicion = st.selectbox('🏷️ Condición', ('Nuevo', 'Usado'))
                precio = st.select_slider('💲 Precio', options=(range(0,300000,500)))
                reviews = st.slider('⭐ Calificación promedio', min_value=1, max_value=5, value=None, step=1)
                color = st.selectbox('🍭 Color', ['Negro',' Blanco', ' Gris', 'Rojo', 'Azul', 'Amarillo', 'Rosa', 'Violeta', 'Marrón'])

            filtros = {'condicion':condicion, 'precio_max':precio, 'review_min':reviews, 'color':color}


    
# Store LLM generated responses
if "messages" not in st.session_state.keys():
    st.session_state.messages = [{"role": "assistant", "content": f"Hola! Soy Marcos-G 😎, cómo puedo ayudarte?"}]

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Function for generating LLM response
def generate_response(email, passwd):
    # Hugging Face Login
    sign = Login(email, passwd)
    cookies = sign.login()
    # Create ChatBot
    chatbot = hugchat.ChatBot(cookies=cookies.get_dict())

    # Add the initial instruction as a system message
    initial_instruction = "Role-play as a shoe salesman and ask all the necessary questions to understand what the customer wants."
    chatbot.conversation.append({"role": "system", "content": initial_instruction})

    return chatbot


# User-provided prompt
user_input = []
if prompt := st.chat_input(disabled=not (hf_email and hf_pass)):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)
        
        user_input.append(prompt)
        
# Generate a new response if last message is not from assistant
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner(f"Pensando...{''':thinking_face:'''}"):
            response = generate_response(prompt, hf_email, hf_pass) 
            st.write(response) 
    message = {"role": "assistant", "content": response}
    st.session_state.messages.append(message)