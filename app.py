import pandas as pd
import os
import numpy as np
import openai
from streamlit_extras.add_vertical_space import add_vertical_space
import streamlit as st
from streamlit_option_menu import option_menu
import Utils
from Utils import set_bg_hack
import streamlit_authenticator as stauth
import yaml
import time
from streamlit_extras.stylable_container import stylable_container


st.set_page_config(page_title="Dubailand Regulator Assistant", page_icon=os.path.join(os.getcwd(), 'icons', 'dt.png'),layout="wide",initial_sidebar_state="auto")

side_bg = 'icons/ai4.png'

with open('config.yaml') as file:
    config = yaml.safe_load(file)



authenticator = stauth.Authenticate(
    config['credentials'],
    config['cookie']['name'],
    config['cookie']['key'],
    config['cookie']['expiry_days'],
    config['preauthorized']
) 

name, authentication_status, username = (authenticator.login())
if authentication_status:
    c11,c22,c33 = st.columns([5,5,1])
    with c33:
        with stylable_container(
                    key="logout_button",
                    css_styles="""
                    button {
                        background-color: #848484;
                        color: white;
                        }
                        """,
                        ):
            st.write('')
elif authentication_status == False:
    st.error('Username/password is incorrect')
    st.stop()
elif authentication_status == None:
    st.warning('Please enter your username and password')
    st.stop()

time.sleep(0.01)

with st.sidebar:

    add_vertical_space(10)
    

    choose = option_menu("Dubai Land Department", ["Home", "Assistant", "Fraud & Anomaly Detection"],
                        icons=["house", "chat-text", "shield-exclamation"],
                        menu_icon="justify", default_index=0,
                        styles={"container": {"padding": "5!important", "background-color": "#fafafa"},
                                "icon": {"color": "black", "font-size": "25px"}, 
                                "nav-link": {"font-size": "16px", "text-align": "left", "margin":"0px", "--hover-color": "#fafafa"},
                                "nav-link-selected": {"background-color": "#29AB87"},
                                }
                        )
    #"Upload PDF",
    #, "dot"
    #, "Fraud & Anomaly Detection", "Real Estate"
    #, "shield-exclamation", "search"
    
    add_vertical_space(30)


def uploadPDF():
    '''
    Call labor definition from labor and part.
    '''
    return Utils.pdfuploader()
def Chatbot(key):
    '''
    Call spare part definition from labor and part.
    '''
    
    Utils.chatbot(key)

def Fraud():
    '''
    Call fraud definition from labor and part.
    '''
    Utils.fraud()

def Eywa2():
    '''
    Call fraud definition from labor and part.
    '''
    Utils.eywa2()

page_names_to_funcs = {"MAIN PAGE": Utils.Main_page,
                    "Upload PDF": uploadPDF,
                    "Online Kılavuz":Chatbot,
                    "Fraud & Anomaly Detection": Fraud,
                    "EYWA2 BENCHMARKING" : Eywa2
                    }

# Call any section from menu.
if choose == "Home":
    page_names_to_funcs["MAIN PAGE"]()
    set_bg_hack(side_bg)
elif choose == 'Upload PDF':
    page_names_to_funcs['Upload PDF']()
elif choose == 'Assistant':
    page_names_to_funcs['Online Kılavuz'](key="kılavuz_key")
elif choose == 'Fraud & Anomaly Detection':
    page_names_to_funcs['Fraud & Anomaly Detection']()
elif choose == 'Real Estate':
    page_names_to_funcs['EYWA2 BENCHMARKING']()    