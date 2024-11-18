#Load libraries
from utils import *
import pandas as pd
import streamlit as st
from openai import OpenAI
import re

#Load item data
#Outputs = list of items, and list of item descriptions
item_data = pd.read_csv('item_descriptions.csv', encoding='cp1252')
items = item_data["item"].tolist()
item_descriptions = item_data["description"].tolist()
links = item_data["link"].tolist()

#Create candidate item prompt string
candidate_prompt_string = create_candidate_item_prompt(items)

#Create item descriptions prompt string
item_descriptions_prompt_string = create_item_descriptions_prompt(item_descriptions)

#Create links prompt string
links_prompt_string = create_links_prompt(links)

#Create embeddings of item descriptions, and turn into dict
#item_embeddings = get_item_embeddings(item_descriptions)
item_embeddings = cache_item_embeddings(item_descriptions)
item_embeddings = dict(zip(items, item_embeddings))

#Load background data and split into chunks
#Output = list of chunked up background text
#background_docs = load_documents('downloaded_pdfs')

#background_docs_chunks = []

#for doc in background_docs:
    #background_docs_chunks.extend(split_text_into_chunks(doc))

#Create embeddings of background documents
#doc_embeddings = get_doc_embeddings(background_docs_chunks)

# Cache background document chunks
background_docs_chunks = cache_document_chunks('downloaded_pdfs')

# Cache document embeddings
doc_embeddings = cache_doc_embeddings(background_docs_chunks)

# Create or load FAISS index
faiss_index = create_faiss_index(doc_embeddings)

#Load user data
test_data = pd.read_csv('testing.csv')
user_email = 'user2@test.com'
user_data = test_data[test_data['user email'] == user_email]

#Extract items user has already interacted with
interacted_items, interaction_prompt_string = extract_interactions(user_data, items)

#Extract profile information
user_profile_prompt_string = f"The user's profile information is as follows: their gender is {user_data.iloc[0]['Gender.1']}, their country is {user_data.iloc[0]['new_country']}, their role is {user_data.iloc[0]['new_role']}, their stakeholder group is {user_data.iloc[0]['Stakeholder Group']}, their organization is {user_data.iloc[0]['Organization']}"

#Extract biographical information
bio_prompt_string = f"In addition the user has provided the following biographical information about themselves (note that in some cases this biographical information is blank, in which case you should disregard it): {user_data.iloc[0]['text']}."

#Get similar items
#similiar_item_prompt_string, top_3 = get_similar_items(interacted_items, item_embeddings)
item_faiss_index, item_list = create_item_faiss_index(item_embeddings)

# Retrieve similar items using FAISS
if interacted_items:
    avg_embedding = average_embeddings(interacted_items, item_embeddings)
    top_3 = retrieve_similar_items(avg_embedding, item_faiss_index, item_list)
    similiar_item_prompt_string = (
        "This is a list of the most similar items to the ones the user has already interacted with. "
        "Items are ranked by their similarity (lower distance = more similar):\n"
        + "\n".join(f"- {item}" for item in top_3)
    )
else:
    similiar_item_prompt_string = (
        "The user has not interacted with any items yet. Recommendations are based on profile information and biography."
    )
    top_3 = []

#Create prompts / messages
task_prompt = read_file('prompts/task_prompt.txt')
few_shot_example1 = read_file('prompts/few_shot_example1.txt')
task_reiteration = read_file('prompts/task_reiteration.txt')
few_shot_answer1 = read_file('prompts/few_shot_answer1.txt')
few_shot_example2 = read_file('prompts/few_shot_example2.txt')
few_shot_answer2 = read_file('prompts/few_shot_answer2.txt')

# Initialize OpenAI client
client = OpenAI(api_key=key)

# Streamlit app setup
#st.title("REDD+ Academy Learning Assistant")
#st.image("title_image.png", width = 60)

#Load the image and convert to Base64
logo_base64 = get_base64_image("avatar_image.png")

#Embed the Base64 image in HTML
st.markdown(f"""
    <div style="display: flex; align-items: center; gap: 15px; margin-bottom: 20px;">
        <img src="data:image/png;base64,{logo_base64}" alt="REDD+ Logo" style="width: 70px; height: auto;">
        <div>
            <h1 style="margin: 0; font-size: 28px; color: #333; padding: 0;">REDD+ Academy Learning Assistant</h1>
            <p style="margin: 3px 0 0 0; color: #666; font-size: 14px; padding: 0; line-height: 1.1;">Developed by the UN-REDD Programme (TEST5)</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

#st.write("""
#This chatbot can provide you with recommendations of learning resources that are tailored to your individual needs. 
#Simply enter a query and hit enter to start the conversation.
#""")

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

    #Provide the descriptions for each candidate item
    {"role": "system", "content": item_descriptions_prompt_string},

    ####Few-shot example 1

    #Provide details of which items the user has already interacted with
    #Provide information about similar items to the one(s) that the user has already interacted with
    #Provide details about the user's profile
    #Provide details about the user's biography
    #Provide details of user group preferences
    {"role": "system", "content": few_shot_example1},
    
    #Reiterate the task
    {"role": "system", "content": task_reiteration},
    {"role": "system", "content": "Remember that you should not recommend items that the user has already interacted with, which are FREL (FAO), NFMS (FAO), PAM (FAO). If there is no relevant information provided by the user or available from their profile or biography, the default should be to recommend the items that are most similar to the items the user has already interacted with, which are Carbon markets for REDD+ (UNEP), REDD+ finance (UNEP), REDD+ safeguards (UNEP)"},
    
    #Provide prompt
    {"role": "user", "content": "Which items would you recommend to me?", "example": True},
    
    #Provide expected output
    {"role": "assistant", "content": few_shot_answer1, "example": True},

    ####Few-shot example 2

    #Provide details of which items the user has already interacted with
    #Provide information about similar items to the one(s) that the user has already interacted with
    #Provide details about the user's profile
    #Provide details about the user's biography
    #Provide details of user group preferences
    {"role": "system", "content": few_shot_example2},

    #Reiterate the task
    {"role": "system", "content": task_reiteration},
    {"role": "system", "content": "Remember that you should not recommend items that the user has already interacted with, which are REDD+ finance (UNEP), Gender and REDD+ (UNDP), REDD+ under UNFCCC (UNDP), Social inclusion and stakeholder engagement (UNDP), REDD+ safeguards (UNEP). If there is no relevant information provided by the user or available from their profile or biography, the default should be to recommend the items that are most similar to the items the user has already interacted with, which are FREL (FAO), National Strategies and Action Plans (UNDP), NFMS (FAO)"},
    
    #Provide prompt
    {"role": "user", "content": "Which items would you recommend to me? I have a particular interest in forest monitoring", "example": True},

    #Provide expected output
    {"role": "assistant", "content": few_shot_answer2, "example": True},

    ####Actual task

    #Provide details of which items the user has already interacted with
    {"role": "system", "content": interaction_prompt_string},

    #Provide information about similar items to the one(s) that the user has already interacted with
    {"role": "system", "content": similiar_item_prompt_string},
    
    #Provide details about the user's profile
    {"role": "system", "content": user_profile_prompt_string},
    
    #Provide details about the user's biography
    {"role": "system", "content": bio_prompt_string},
    
    #Reiterate the task
    {"role": "system", "content": task_reiteration},
    {"role": "system", "content": f"Remember that you should not recommend items that the user has already interacted with, which are {interacted_items}. If there is no relevant information provided by the user or available from their profile or biography, the default should be to recommend the items that are most similar to the items the user has already interacted with, which are {top_3}"},
    
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

#avatar_path = 'avatar_image.png'

#for message in st.session_state.visible_messages:
    #with st.chat_message(message["role"], avatar=avatar_path if message["role"] == "assistant" else None):
        #st.markdown(message["content"])

# Input prompt with st.chat_input
if user_input := st.chat_input("You:"):
    # Add user message to visible conversation history
    st.session_state.visible_messages.append({"role": "user", "content": user_input})
    
    # Display user message in the chat
    with st.chat_message("user"):
        st.markdown(user_input)

    # Retrieve relevant text based on the user's input
    #retrieved_text = retrieve_documents(user_input, doc_embeddings, background_docs_chunks)

    # Retrieve relevant text using FAISS
    retrieved_text = retrieve_documents(user_input, faiss_index, background_docs_chunks)

    # Add retrieved context to the background conversation history
    context = f"Context: {retrieved_text}\n\nUser Query: {user_input}"
    st.session_state.background_messages.append({"role": "system", "content": context})

    # Combine background and visible messages for the API call
    combined_messages = st.session_state.background_messages + st.session_state.visible_messages

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