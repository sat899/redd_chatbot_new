#Load libraries
from utils import *
import pandas as pd
import streamlit as st
from openai import OpenAI
import re

#Load item data
#Outputs = list of items, list of item descriptions, list of links
item_data = pd.read_csv('item_descriptions.csv', encoding='utf-8')
items = item_data["item"].tolist()
item_descriptions = item_data["description"].tolist()
item_descriptions = [str(description) for description in item_descriptions]
links = item_data["link"].tolist()

#Create candidate item prompt string
candidate_prompt_string = create_candidate_item_prompt(items)

#Create links prompt string
links_prompt_string = create_links_prompt(links)

#Create embeddings of item descriptions, and turn into dict
item_embeddings = get_item_embeddings(item_descriptions)
item_embeddings = cache_item_embeddings(item_descriptions)
item_embeddings = dict(zip(items, item_embeddings))

# Create or load item FAISS index
item_faiss_index, item_list = create_item_faiss_index(item_embeddings)

# Load and cache background document chunks
background_docs_chunks = cache_document_chunks('downloaded_pdfs')

# Cache document embeddings
doc_embeddings = cache_doc_embeddings(background_docs_chunks)

# Create or load document FAISS index
doc_faiss_index = create_doc_faiss_index(doc_embeddings)

#Create prompts / messages
task_prompt = read_file('prompts/task_prompt.txt')
recommendation_answer = read_file('prompts/recommendation_answer.txt')
summarisation_answer = read_file('prompts/summarisation_answer.txt')

# Initialize OpenAI client
client = OpenAI(api_key=key)

#Load the image and convert to Base64
logo_base64 = get_base64_image("title_image.png")

#Embed the Base64 image in HTML
st.markdown(f"""
    <div style="display: flex; align-items: center; gap: 15px; margin-bottom: 20px;">
        <img src="data:image/png;base64,{logo_base64}" alt="REDD+ Logo" style="width: 70px; height: auto;">
        <div>
            <h1 style="margin: 0; font-size: 28px; color: #333; padding: 0;">REDD+ Academy Learning Assistant</h1>
            <p style="margin: 3px 0 0 0; color: #666; font-size: 14px; padding: 0; line-height: 1.1;">Developed by the UN-REDD Programme</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

# Initialize Streamlit session state variables
if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = "gpt-4o-mini"

# Initialize session state for background conversation history
if 'background_messages' not in st.session_state:
    st.session_state.background_messages = [

    #Provide information about the task, the role the LLM should assume and the expected output
    {"role": "system", "content": task_prompt},
        
    #Provide the candidate items that can be recommended
    {"role": "system", "content": candidate_prompt_string},

    #Provide the links for each candidate item
    {"role": "system", "content": links_prompt_string},

    ###Few-shot example - recommendation

    #Provide prompt
    {"role": "user", "content": "This is an example of a recommendation task / question: I am looking for resources related to REDD+ finance", "example": True},
    
    #Provide expected output
    {"role": "assistant", "content": f"this is an example of a good answer to this question{recommendation_answer}", "example": True},

    ###Few-shot example - summarisation

    #Provide prompt
    {"role": "user", "content": "This is an example of a summarisation task / question: Tell me about how to develop a nested system for REDD+", "example": True},
    
    #Provide expected output
    {"role": "assistant", "content": f"this is an example of a good answer to this question{summarisation_answer}", "example": True},
    
  ]
    
# Initialize session state for visible conversation history
if 'visible_messages' not in st.session_state:
    st.session_state.visible_messages = []
    # Generate the greeting once and add it to visible messages
    greeting = "Hello! Welcome to the REDD+ Academy Website. How can I help you today?"
    st.session_state.visible_messages.append({"role": "assistant", "content": greeting})

# Display all visible messages
for message in st.session_state.visible_messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Input prompt with st.chat_input
if user_input := st.chat_input("You:"):
    # Add user message to visible conversation history
    st.session_state.visible_messages.append({"role": "user", "content": user_input})
    
    # Display user message in the chat
    with st.chat_message("user"):
        st.markdown(user_input)

    # Retrieve relevant items based on the user's input
    retrieved_items = retrieve_similar_items(user_input, item_faiss_index, item_list)

    # Print the retrieved items for testing
    #print(f"Retrieved Items:\n{retrieved_items}\n")

    # Format the retrieved items into a prompt-friendly string
    retrieved_items_prompt = "The most relevant items based on the user's query appear to be:\n" + "\n".join(f"- {item}" for item in retrieved_items)

    # Retrieve relevant docs using FAISS
    retrieved_text = retrieve_documents(user_input, doc_faiss_index, background_docs_chunks)

    # Print the retrieved context for testing
    #print(f"Retrieved Context:\n{retrieved_text}\n")

    # Add retrieved context and items to the background conversation history
    context = f"{retrieved_items_prompt}\n\nContext: {retrieved_text}\n\nUser Query: {user_input}"
    st.session_state.background_messages.append({"role": "system", "content": context})

    # Add a final reminder prompt for the bot
    final_reminder = "Reminder: you should do one of two tasks, either recommendation or summarisation. Based on the user's query, decide which task to do and refer to the previous examples for examples of how to respond. For recommendation tasks any links you recommend should come from the list shared previously. Please prioritise the links to learning journals and PDFs on the howspace webiste and the un-redd.org website. Specifically do not recommend any links starting with unredd.net"
    st.session_state.background_messages.append({"role": "system", "content": final_reminder})

    # Combine background and visible messages for the API call
    combined_messages = st.session_state.background_messages + st.session_state.visible_messages

    # Print combined messages for debugging
    print("Combined Messages:")
    for msg in combined_messages:
        print(f"Role: {msg['role']}, Content: {msg['content']}\n")

    # Generate and stream response from OpenAI
    with st.chat_message("assistant", avatar = None):
        message_placeholder = st.empty()
        full_response = ""
        
        stream = client.chat.completions.create(
            model=st.session_state["openai_model"],
            messages=[
                {"role": m["role"], "content": m["content"]}
                for m in combined_messages
            ],
            stream=True,
        )
        
        for chunk in stream:
            if chunk.choices[0].delta.content is not None:
                full_response += chunk.choices[0].delta.content
                
                #Split the full response into paragraphs
                paragraphs = re.split(r'\n\s*\n', full_response.strip())
                
                #Join paragraphs with double line breaks and display
                formatted_response = "\n\n".join(paragraphs)
                formatted_response = formatted_response.replace("\n", "\n\n")
                message_placeholder.markdown(formatted_response + "▌")
        
        # Remove the blinking cursor and display the final response
        message_placeholder.markdown(formatted_response)

    # Add assistant message to visible conversation history
    st.session_state.visible_messages.append({"role": "assistant", "content": formatted_response})

    # Force a rerun to update the display with the new message
    st.rerun()