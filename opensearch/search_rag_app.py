import streamlit as st #all streamlit commands will be available through the "st" alias
from search_rag import invoke


st.set_page_config(page_title="RAG with OpenSearch", layout="wide") #HTML title
st.title("RAG with OpenSearch") #page title



# if 'vector_index' not in st.session_state: #see if the vector index hasn't been created yet
#     with st.spinner("Indexing document..."): #show a spinner while the code in this with block runs
#         st.session_state.vector_index = glib.get_index() #retrieve the index through the supporting library and store in the app's session cache



input_text = st.text_input("Ask a question about content from changi-airport.com") #display a multiline text box with no label
go_button = st.button("Go", type="primary") #display a primary button



if go_button: #code in this if block will be run when the button is clicked
    
    with st.spinner("Working..."): #show a spinner while the code in this with block runs
        response_content = invoke(input_text)
        
        st.write(response_content) #using table so text will wrap
        
        
        # raw_embedding = glib.get_embedding(input_text)
        
        # with st.expander("View question embedding"):
        #     st.json(raw_embedding)