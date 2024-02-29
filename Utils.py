import pandas as pd
import os
import numpy as np
import openai
from openai.embeddings_utils import distances_from_embeddings,cosine_similarity
from streamlit_extras.add_vertical_space import add_vertical_space
import streamlit as st
from streamlit_extras.colored_header import colored_header
from streamlit_chat import message
import tiktoken
import json
import PyPDF2
import base64
from openai.embeddings_utils import distances_from_embeddings
from streamlit_feedback import streamlit_feedback
from streamlit_extras.stylable_container import stylable_container




openai.api_type = "azure"
openai.api_base = "https://poc-aoi-ce.openai.azure.com/"
openai.api_version = "2023-07-01-preview"
openai.api_key = os.getenv("74c28ca802a34201b6430f4281c1142f")



def set_bg_hack(main_bg):
    # set bg name
    main_bg_ext = "png"
        
    st.markdown(
         f"""
         <style>
         .stApp {{
             background: url(data:image/{main_bg_ext};base64,{base64.b64encode(open(main_bg, "rb").read()).decode()});
             background-size: cover
         }}
         </style>
         """,
         unsafe_allow_html=True
     )
   
def clean_and_convert_arabic(text):
    cleaned_text = text.encode('latin1').decode('utf-8')
    return cleaned_text

def add_bg_from_local(image_file):
    '''
    Add background from local. Nested function
    '''
    with open(f'icons/{image_file}', "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    st.markdown(f"""
                <style>.stApp {{
                background-image: url(data:image/{"png"};base64,{encoded_string.decode()});
                background-size: cover
                              }}
                </style>
                 """,
                unsafe_allow_html=True
                )

def Main_page():
    '''
    Creating landing page.
    '''
    #add_bg_from_local('deneme.png')

    
def num_tokens_from_messages(messages, model="gpt-3.5-turbo-0613"):
    """Return the number of tokens used by a list of messages."""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        print("Warning: model not found. Using cl100k_base encoding.")
        encoding = tiktoken.get_encoding("cl100k_base")
    if model in {
        "gpt-3.5-turbo-0613",
        "gpt-3.5-turbo-16k-0613",
        "gpt-4-0314",
        "gpt-4-32k-0314",
        "gpt-4-0613",
        "gpt-4-32k-0613",
        "gpt-3.5-turbo-1106",
        "gpt-4-1106-preview"
        }:
        tokens_per_message = 3
        tokens_per_name = 1
    elif model == "gpt-3.5-turbo-0301":
        tokens_per_message = 4  # every message follows <|start|>{role/name}\n{content}<|end|>\n
        tokens_per_name = -1  # if there's a name, the role is omitted
    elif "gpt-3.5-turbo" in model:
        print("Warning: gpt-3.5-turbo may update over time. Returning num tokens assuming gpt-3.5-turbo-0613.")
        return num_tokens_from_messages(messages, model="gpt-3.5-turbo-0613")
    elif "gpt-4" in model:
        print("Warning: gpt-4 may update over time. Returning num tokens assuming gpt-4-0613.")
        return num_tokens_from_messages(messages, model="gpt-4-0613")
    else:
        raise NotImplementedError(
            f"""num_tokens_from_messages() is not implemented for model {model}. See https://github.com/openai/openai-python/blob/main/chatml.md for information on how messages are converted to tokens."""
        )
    num_tokens = 0
    for message in messages:
        num_tokens += tokens_per_message
        for key, value in message.items():
            num_tokens += len(encoding.encode(value))
            if key == "name":
                num_tokens += tokens_per_name
    num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
    return num_tokens


def create_context(
    question, df, max_len=1800, size="ada",num_contexts=3, similarity_threshold=0.3
):


    # Get the embeddings for the question
    q_embeddings = openai.Embedding.create(input=question, engine='dev-ada-002')['data'][0]['embedding']

    # Get the distances from the embeddings
    df['distances'] = distances_from_embeddings(q_embeddings, df['embeddings'].values, distance_metric='cosine')

   # Filter the DataFrame to get only the closest contexts with similarity > similarity_threshold
    closest_filtered = df[df['distances'] < similarity_threshold].nsmallest(num_contexts, 'distances')

    # Get the closest contexts and their cosine similarity values
    closest_contexts = closest_filtered['text'].tolist()

    return closest_contexts



    

def answer_question(
    df,
    model="gpt-4-1106-preview",
    question="Create a English context for a English question by finding the most similar context from the data content frame.",  ##burası kaldırılabilir
    max_len=2000,
    size="ada",
    debug=False,
    max_tokens=1000,
    stop_sequence=None
):


    context = create_context(
        question,
        df,
        max_len=max_len,
        size=size,
    )
    # If debug, print the raw model response
    if debug:
        print("Context:\n" + context)
        print("\n\n")



    try:
        
        # Create a completions using the question and context

        prompt =f"""
                You are a virtual assistant on the Government of Dubai's Dubai Land Department website. 
                You will be asked questions about rules and regulations, and you must give the most appropriate, meaningful, detailed and human-like answer from the context given to you. 
                These contents will be on the following lines, just after the 'Content' heading. 
                Find the context closest to the question I asked, inspired by the 'Content' I will give, and write a meaningful answer. 
                As much as possible, do not obtain data from any source other than the given content.
                I will provide you with content that may answer your question right next to the 'content' heading. 
                Right next to the 'Question' heading will be the question asked by the user. 
                I want you to find the most appropriate 'Answer' to the question from 'Content'.
                Here is the content and the question asked: \n\n 'Content': {context}\n\n---\n\n 'Question': {question}\n\n 
                Do not say 'Answer' at the beginning when answering.
                Answering questions that contain things like profanity, violence, sexism, sexuality and slang. "I can't answer that question." also.
                """

        st.session_state['messages'].append({"role": "user", "content": prompt})



        response = openai.ChatCompletion.create(
            
            #prompt=f"Soruyu verilen bağlama göre yanıtla. Eğer soru bağlama göre yanıtlanamıyorsa \"Bu konu hakkında bilgim yok, detaylı bilgi için Audi yetkili servisleri ile iletişime geçiniz.\"\n\nİçerik: {context}\n\n---\n\nSoru: {question}\nCevap:",
            temperature=0.05,
            max_tokens=max_tokens,
            top_p=0.9,
            frequency_penalty=0,
            presence_penalty=0,
            stop=stop_sequence,
            engine=model,
            messages=st.session_state['messages'][-4:]


        )
        
        message_response = response["choices"][0]["message"]["content"]
        st.session_state['messages'].append({"role": "assistant", "content": message_response})

        return message_response
    except Exception as e:
        print(e)
        return ""
    

def ask(df,query):
    message_response = answer_question(df,question=query)
    return message_response




def input_change():
    
    output = ask(df,st.session_state.input)
    # store the output 
    st.session_state.past.append(st.session_state.input)
    st.session_state.generated.append(output)
    st.session_state.input = ""

def input_change3(text):
    
    output = ask(df,text)
    # store the output 
    st.session_state.past.append(text)
    st.session_state.generated.append(output)
    st.session_state.input = ""

def input_change2(text1):

    output = ask(df,text1)

    st.session_state.past.append(text1)
    st.session_state.generated.append(output)
    st.session_state.input = ""



def remove_newlines(serie):
    serie = serie.str.replace('\n', ' ')
    serie = serie.str.replace('\\n', ' ')
    serie = serie.str.replace('  ', ' ')
    serie = serie.str.replace('  ', ' ')
    serie = serie.str.replace('', ' ')
    serie = serie.str.replace('-', ' ')
    serie = serie.str.replace('', '')
    return serie

def split_into_many(text,tokenizer, max_tokens = 500):

    # Split the text into sentences
    sentences = text.split('. ')

    # Get the number of tokens for each sentence

    n_tokens = [len(tokenizer.encode(" " + sentence)) for sentence in sentences]

    chunks = []
    tokens_so_far = 0
    chunk = []

    # Loop through the sentences and tokens joined together in a tuple
    for sentence, token in zip(sentences, n_tokens):

        # If the number of tokens so far plus the number of tokens in the current sentence is greater
        # than the max number of tokens, then add the chunk to the list of chunks and reset
        # the chunk and tokens so far
        if tokens_so_far + token > max_tokens:
            chunks.append(". ".join(chunk) + ".")
            chunk = []
            tokens_so_far = 0

        # If the number of tokens in the current sentence is greater than the max number of
        # tokens, go to the next sentence
        if token > max_tokens:
            continue

        # Otherwise, add the sentence to the chunk and add the number of tokens to the total
        chunk.append(sentence)
        tokens_so_far += token + 1

    return chunks

def pdfuploader():
    pdfFileObj = st.file_uploader("Upload PDF", type="pdf")

    if pdfFileObj:
        with open(f'docs/{pdfFileObj.name}', mode='wb') as w:
            w.write(pdfFileObj.getvalue())
        with st.spinner(text='File embedding in progress!'):
            openai.api_key = ("74c28ca802a34201b6430f4281c1142f")
            #pdfFileObj = open('docs/Passat, Passat Variant, Passat Alltrack_052016.pdf', 'rb')
            
            pdfReader = PyPDF2.PdfReader(pdfFileObj)

            output = []
            for i in range(0,len(pdfReader.pages)):
                page = pdfReader.pages[i]
                read_list = {'page_num' : i,
                            'text' : page.extract_text()}
                output.append(read_list)

            df = pd.DataFrame(filter(None,output) )
            df.text = remove_newlines(df.text)
            df['text'] = df['text'].astype(str)
            df['text'] = df['text'].apply(lambda x: x.encode('utf-8').decode('latin-1'))
            pdfFileObj_name = pdfFileObj.name.split('.')[0]
            # closing the pdf file object
            pdfFileObj.close()

            df.to_csv(f'Scraped/{pdfFileObj.name.split(".")[0]}.csv')
            max_tokens = 500
            # Load the cl100k_base tokenizer which is designed to work with the ada-002 model
            tokenizer = tiktoken.get_encoding("cl100k_base")

            df = pd.read_csv(f'Scraped/{pdfFileObj.name.split(".")[0]}.csv', index_col=0)
            df.columns = ['page_num', 'text']

            # Tokenize the text and save the number of tokens to a new column
            df['n_tokens'] = df.text.apply(lambda x: len(tokenizer.encode(str(x))))

            shortened = []

            # Loop through the dataframe
            for row in df.iterrows():

                # If the text is None, go to the next row
                if row[1]['text'] is None:
                    continue

                # If the number of tokens is greater than the max number of tokens, split the text into chunks
                if row[1]['n_tokens'] > max_tokens:
                    shortened += split_into_many(row[1]['text'],tokenizer)

                # Otherwise, add the text to the list of shortened texts
                else:
                    shortened.append( row[1]['text'] )



            df = pd.DataFrame(shortened, columns = ['text'])
            df['n_tokens'] = df.text.apply(lambda x: len(tokenizer.encode(str(x))))


            df['embeddings'] = df.text.apply(lambda x: openai.Embedding.create(input=str(x), engine='dev-ada-002')['data'][0]['embedding'])
            f = open('EmbedFiles.json')
            jsn = json.load(f)
            jsn[0][f'{pdfFileObj_name}'] = f'EmbeddingFiles/{pdfFileObj_name}.csv'
            with open("EmbedFiles.json", "w") as jsonFile:
                json.dump(jsn, jsonFile)

            df.to_csv(f'EmbeddingFiles/{pdfFileObj_name}.csv',index=False)
            print('Embedding process finished!')



def chatbot(key="ch_key"):
    import json
    aaa = {
           'Emirates Book Valuation Standards': 'valution_merged',
           'Jointly Owned Property in the Emirate of Dubai': 'jointlyowned_merged',
           'License Circulars': 'license_merged',
           'Real Estate Legislation': 'legislation_merged', 
           'Tenancy Guide': 'tenancyguideen', 
           'Global Corparate Real Estate Guide':'Global Corparate Real Estate Guide'

    }
    st.markdown("<h3 style='text-align: center; color: #000000;'>Welcome 🤗</h3>", unsafe_allow_html=True)
    #st.markdown("<h5 style='text-align: center; color: #000000;'>Chat with the AI-Powered Dubai Land Department virtual assistant and browse rules & regulations stress free!</h5>", unsafe_allow_html=True)
    st.markdown("<h5 style='text-align: center; color: #000000;'>Please select a section </h5>", unsafe_allow_html=True)

    c1,c2,c3 = st.columns([1,1,1])

    
    
    pdf = c2.selectbox('xx',[k for k,v in aaa.items()],key=key,label_visibility="collapsed")


    
    
    if pdf != ' ': 
        if pdf == 'Emirates Book Valuation Standards':

            quickreply_q1="What information must a valuer inform their client of before accomplishing the valuation task?"
            quickreply_q2="What should be included in the notes when inspecting a property for valuation?"
            quickreply_q3="What should be attached to the property valuation report?"
            quickreply_q4="What equipment is required for inspecting and examining a property?"

        if pdf == 'Jointly Owned Property in the Emirate of Dubai':

            quickreply_q1="Who is responsible for the costs and expenses associated with a plot before and after it becomes Jointly Owned Property?"
            quickreply_q2="What is the Developer's liability if the information in a disclosure statement is found to be materially inaccurate or incomplete?"
            quickreply_q3="What are the consequences if a Consumer does not provide a copy of a statement in accordance with Article (4)?"
            quickreply_q4="What happens if a Developer fails to provide a disclosure statement as required by Article (4)?"

        if pdf == 'License Circulars':

            quickreply_q1="What is required of real estate agents when renewing their residency visa?"
            quickreply_q2="What actions must a Time Share Company take according to the circular from RERA?"
            quickreply_q3="What are the consequences for real estate brokers who do not obtain a real estate broker card by the specified deadline?"
            quickreply_q4="What is the new system introduced for Real Estate Brokers Offices, and what are they required to do?"
        
        if pdf == 'Real Estate Legislation':

            quickreply_q1="What is the role of the Real Estate Regulatory Agency (RERA)?"
            quickreply_q2="What are some other significant laws and decrees related to real estate in Dubai?"
            quickreply_q3="Relationship between landlords and tenants in Dubai - Law No. (33) of 2008 entail"
            quickreply_q4="Regulation of mortgages in the Emirate of Dubai - Law No. (14) of 2008"


        if pdf == 'Tenancy Guide':

            quickreply_q1="What documents are required for a representative of the owner to manage a property?"
            quickreply_q2="What are the obligations of an owner who personally manages their property?"
            quickreply_q3="What are the requirements for an owner delegating property management to a company?"
            quickreply_q4="How should the contractual relationship between landlord and tenant be regulated?"

        
        if pdf == 'Global Corparate Real Estate Guide':

            quickreply_q1="What laws govern real estate transactions?"
            quickreply_q2="What is included in the term “real estate”?"
            quickreply_q3="What is the land registration system?"
            quickreply_q4="What rights over real property are required to be registered?"

        f = open("EmbedFiles.json",)
        dictofpdf = json.load(f)
        global df
        df = pd.read_csv(dictofpdf[0][aaa[pdf]])

        openai.api_key = ("74c28ca802a34201b6430f4281c1142f")
        API_ENDPOINT = "https://poc-aoi-ce.openai.azure.com/openai/deployments/gpt-4-1106-preview/chat/completions?api-version=2023-07-01-preview"

        df['embeddings'] = df['embeddings'].apply(eval).apply(np.array)
        

        if 'messages' not in st.session_state:
            

            st.session_state['messages'] = [
                {"role": "system", 
                 "content": """
                You are a virtual assistant on the Government of Dubai's Dubai Land Department website. 
                You will be asked questions about rules and regulations, and you must give the most appropriate, meaningful, detailed and human-like answer from the context given to you. 
                These contents will be on the following lines, just after the 'Content' heading. 
                Find the context closest to the question I asked, inspired by the 'Content' I will give, and write a meaningful answer. 
                As much as possible, do not obtain data from any source other than the given content.
                I will provide you with content that may answer your question right next to the 'content' heading. 
                Right next to the 'Question' heading will be the question asked by the user. 
                I want you to find the most appropriate 'Answer' to the question from 'Content'.
                Here is the content and the question asked: \n\n 'Content': {context}\n\n---\n\n 'Question': {question}\n\n 
                Do not say 'Answer' at the beginning when answering.
                Answering questions that contain things like profanity, violence, sexism, sexuality and slang. "I can't answer that question." also.
                """
                 }
            ]
        # Storing the chat
        if 'generated' not in st.session_state:
            st.session_state['generated'] = []

        if 'past' not in st.session_state:
            st.session_state['past'] = []


        #index = construct_index("docs")
        chat_placeholder = st.empty()

        with chat_placeholder.container():

            add_vertical_space(1)

            c111,c222,c333,c444 = st.columns(4)
            with c111:
                with stylable_container(
                    key="green_button1",
                    css_styles="""
                    button {
                        background-color: #90CF8E;
                        color: white;
                        height:130px;
                        }
                        """,
                        ):
                    if st.button(quickreply_q1,use_container_width=True,key="1key"):
                        input_change2(quickreply_q1)
            with c222:
                with stylable_container(
                    key="green_button2",
                    css_styles="""
                    button {
                        background-color: #90CF8E;
                        color: white;
                        height:130px;
                        }
                        """,
                        ):
                    if st.button(quickreply_q2,use_container_width=True,key="2key"):
                        input_change2(quickreply_q2)
            with c333:
                with stylable_container(
                    key="green_button3",
                    css_styles="""
                    button {
                        background-color: #90CF8E;
                        color: white;
                        height:130px;
                        }
                        """,
                        ):
                    if st.button(quickreply_q3,use_container_width=True,key="3key"):
                        input_change2(quickreply_q3)
            
            with c444:
                with stylable_container(
                    key="green_button4",
                    css_styles="""
                    button {
                        background-color: #90CF8E;
                        color: white;
                        height:130px;
                        position:'fixed'
                        }
                        """,
                        ):
                    if st.button(quickreply_q4,use_container_width=True,key="4key"):
                        input_change2(quickreply_q4)
                        
            add_vertical_space(2)

            message('Hi! 👋', is_user=False,logo="https://i.ibb.co/TPtdzjT/user.jpg")
            message('How can I assist you?', is_user=False,logo="https://i.ibb.co/TPtdzjT/user.jpg")
            c11,c22,c33 = st.columns([4,3,3])
            with c11: 
                feedback = streamlit_feedback(feedback_type='thumbs',  #"thumbs",faces
                                        optional_text_label="Please give me feedback.",
                                        #max_text_length= 100,
                                        key='baslangic' + '_feedback', 
                                        align='flex-start') #thumbs  align='flex-end'
                feedback


            if st.session_state['generated']:
                for i in range(len(st.session_state['generated'])):
                    message(st.session_state['past'][i], is_user=True, key=str(i) + '_user',logo="https://i.ibb.co/kmMcZbF/user2.jpg")
                    message(st.session_state["generated"][i], is_user=False, key=str(i),logo="https://i.ibb.co/TPtdzjT/user.jpg") 
                    c11,c22,c33 = st.columns([4,3,3])
                    with c11: 
                        feedback = streamlit_feedback(feedback_type='thumbs',  #"thumbs",faces
                                              optional_text_label="Please give me feedback.",
                                              #max_text_length= 100,
                                              key=str(i) + '_feedback', 
                                              align='flex-start') #thumbs  align='flex-end'
                        feedback
            
            
                    

        with st.container():
            add_vertical_space(5)
            c11,c22 = st.columns([2,15])
            with c11:
                reset_button_key = "reset_button"
                with stylable_container(
                    key="green_button",
                    css_styles="""
                    button {
                        background-color: #29AB87;
                        color: white;
                        }
                        """,
                        ):
                    if st.button("Clear chat :broom: ", use_container_width=True,key=reset_button_key, on_click=clear_message_history):
                        st.session_state['past'].clear()
                        st.session_state['generated'].clear()
            with c22:
                st.text_input("Please write your question: ", key="input",on_change=input_change,label_visibility="collapsed")
            #with c33:
               # with stylable_container(
                    #key="green_button",
                    #css_styles="""
                   # button {
                      #  background-color: #29AB87;
                      #  color: white;
                      #  }
                     #   """,
                      #  ):
                   # if st.button("Ask :rocket: ",key="enter_button_key",use_container_width=True):
                    #    input_change2(str(aa))



def clear_message_history():
    st.session_state['past'].clear()
    st.session_state['generated'].clear()

def fraud():
    from poc_son_functions import prepare_data, domain_knowledge

    def remove_duplicates(lst):
        seen = set()  # Use a set to track seen elements
        new_lst = []  # List to store elements without duplicates
        for element in lst:
            if element not in seen:
                new_lst.append(element)
                seen.add(element)
        return new_lst

    def add_bg_from_local(image_file):
        '''
        Add background from local. Nested function
        '''
        with open(f'icons/{image_file}', "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read())
        st.markdown(f"""
                    <style>.stApp {{
                    background-image: url(data:image/{"jpg"};base64,{encoded_string.decode()});
                    background-size: cover}}
                    </style>
                    """,
                    unsafe_allow_html=True
                    )

    def display_score_with_color_bar(score, identifier):
        score_percentage = score * 100
        st.markdown(f"""
        <style>
        .color-bar-container-{identifier} {{
            width: 100%;
            height: 30px;
            background: linear-gradient(to right, #00ff00, #ffff00, #ff0000);
            position: relative;
        }}
        .score-pointer-{identifier} {{
            position: absolute;
            top: 0;
            left: {score_percentage}%;
            width: 2px;
            height: 30px;
            background-color: #000;
            transform: translateX(-50%);
        }}
        .score-label-{identifier} {{
            position: absolute;
            top: 35px;
            left: {score_percentage}%;
            transform: translateX(-50%);
            font-size: 16px;
            color: #000;
        }}
        </style>
        <div class="color-bar-container-{identifier}">
            <div class="score-pointer-{identifier}"></div>
        </div>
        <div class="score-label-{identifier}">Score: {score_percentage:.2f}%</div>
        """, unsafe_allow_html=True)

    side_bg = 'bb8_v1.jpg'
    add_bg_from_local(side_bg)

    st.info("Fetching data, please wait...")

    @st.cache_data
    def load_data():
        return pd.read_feather('fraud_data/dubai_res.feather')

    @st.cache_data
    def fetching_data():
        return prepare_data()

    # Initialize data
    dummy_data_lof = load_data()
    dummy_data_domain = fetching_data()

    st.success("Data successfully loaded!")

    with open('fraud_data/feature_names.json', 'r') as file:
        features_names = json.load(file)

    st.markdown("""
    <div style="font-size: 14px;">
        
    ###### Objective
    <span style="font-size: 14px;">To identify and analyze similar property listings based on specific criteria to detect potential fraud.</span>

    ###### Criteria for Similarity
    <span style="font-size: 14px;">
    1. Property Matching: Listings with the same property according to name.<br>
    2. Size Variance: Within ±1% size difference.<br>
    3. Price Range: Within ±10% price difference.<br>
    4. Bedroom Count: Same number of bedrooms.<br>
    5. Publishing Window: Listed within ±28 days of each other.<br><br> <!-- Burada ekstra boşluk ekliyoruz -->
    </span>
                    
    ###### Methodology
    <span style="font-size: 14px;">
    - Properties are sorted by their publish date.<br>
    - Each property is compared against these criteria to find similarities with others.<br>
    - A unique number is assigned to each property to denote similar groups.<br>
    - A fraud score is calculated based on the uniqueness of the agent within the context of these similarities.
    </span>

    </div>
    """, unsafe_allow_html=True)

    
    # Retrieve data based on ID (for demonstration purposes)
    def get_data_by_id_lof(data, selected_url):

        return data[data['source_url'] == selected_url].LOF_Score.values[0], data[data['source_url'] == selected_url].custom_scaled_score.values[0], data[data['source_url'] == selected_url].shap_top3.values[0]
    
    # User input for ID
    selected_url = st.text_input("Enter URL")
    selected_url = selected_url.strip()
    
    # Display Explanation and Recommendation when the button is clicked
    if st.button("Get Result", key='explanation_button'):
        if (selected_url in dummy_data_domain['source_url'].values) & (selected_url in dummy_data_lof['source_url'].values):
            score, frequency, agent_name,office_name = domain_knowledge(dummy_data_domain, selected_url)
            if score == 0:
                result = "Non Fraud"
            else:
                result = "Fraud"
            if str(agent_name) == 'nan':
                agent_name = 'Unknown'
            if str(office_name) == 'nan':
                office_name = 'Unknown'
            st.markdown(f"<span class='red-text'>**Fraud Result:**</span> For the given URL, the associated agent's name is {agent_name} and the corresponding office name is {office_name}. A score of {score*100}% indicates a potential {result}. The property is represented by {frequency}. agent.", unsafe_allow_html=True)
            display_score_with_color_bar(score, "fraud")

            st.markdown('<hr>', unsafe_allow_html=True)

            result, scaled_score,shap_top3 = get_data_by_id_lof(dummy_data_lof,selected_url)
            if result == -1:
                result = 'Potential Anomaly'
            else:
                result = 'Non Anomaly'

            shap_top3 = eval(shap_top3)
            shap_top3 = remove_duplicates(shap_top3)[0:3]
            shap_features_list_md = "\n".join([f"- {features_names[name]}" for name in shap_top3])

            # Now, integrate this Markdown-formatted list into your existing Markdown output
            st.markdown(f"""
            <span class='red-text'>**Anomaly Result:**</span> It is indicated as a <strong>{result}</strong>. The most important features in making this decision are:

            \n{shap_features_list_md}
            """, unsafe_allow_html=True)

            display_score_with_color_bar(scaled_score/100, "anomaly")
            
        else:
            st.error("Invalid URL. Please enter a valid URL.")
                
def eywa2():
    df = pd.read_csv("eywa_data/Benchmarking.csv")
    df2 = pd.read_csv("eywa_data/grouped_benchmark.csv")
    st.title("Real Estate Benchmarking")

    property = st.selectbox(
    "Select the property",
    df['PropertyName'].unique(),
    placeholder="Select the property...",
    )

    if st.button("Show Data"):
        selected_cols = ["count", "avg_price_per_size", "med_price_per_size", "avg_price", "med_price"]

        df_filtered = df[df['PropertyName'] == property][selected_cols]
        df_filtered.rename(columns={"count": "Number of Transactions",
                                        "avg_price_per_size": "Average Price per Size",
                                        "med_price_per_size": "Median Price per Size", 
                                        "avg_price": "Average Price", 
                                        "med_price": "Median Price"}, inplace=True)
        st.header("Sale")
        st.dataframe(df_filtered, hide_index = True)
        df_filtered2 = df2[df2['PropertyName'] == property].sort_values("RoomType").drop("PropertyName", axis=1)
        df_filtered2 = df_filtered2[~df_filtered2.actual.isin(["Not Available"])]
        df_filtered2.rename(columns = {"actual": "Average Size in m²",
                                        "count": "Room Count",
                                        "actual_min": "Average Minimum Size in m²",
                                        "actual_max": "Average Maximum Size in m²"}, inplace=True)
        st.header("Supply")    
        st.dataframe(df_filtered2, hide_index = True)
        st.markdown("""- Approach:
        AI-based Constrained Clustering""")
        st.markdown("""- Used Property Characteristics in Clustering:
        Location
        Demand and Supply
        Price per Square Meter
        Amenities
        Unit Sizes (e.g., studios, 2+1 units)""")
        st.markdown("""- Project-Specific Constraints:
        Area Size
        Minimum and Maximum Price per Square Meter""")
        st.header("Recommendation")
        x = df2[pd.to_numeric(df2['actual'], errors='coerce').notnull()]
        x[['actual', 'actual_min', 'actual_max', 'count']] = x[['actual', 'actual_min', 'actual_max', 'count']].astype('float')
        y = x.groupby("RoomType")[['actual', 'actual_min', 'actual_max', 'count']].mean().reset_index()
        rec_df = pd.DataFrame()
        rec_df['RoomType'] = y['RoomType']
        y = y[['actual', 'actual_min', 'actual_max', 'count']].apply(lambda x: round(x, 1))
        y['count'] = y['count'].apply(lambda x: round(x, 0))
        rec_df[["Recommended Size in m²", "Recommended Min. Size in m²", "Recommended Max. Size in m²", "Recommended Room Count"]] = y[['actual', 'actual_min', 'actual_max', 'count']]
        rec_df = rec_df[~rec_df.RoomType.isin(['5 B/R', '8 B/R'])]
        rec_df.loc[rec_df.RoomType == "1 B/R", "Recommended Room Count"] = 43
        rec_df.loc[rec_df.RoomType == "Studio", "Recommended Room Count"] = 20
        st.dataframe(rec_df, hide_index = True)