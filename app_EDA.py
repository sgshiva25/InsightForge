# Import required libraries
import os
import re
import copy
from apikey import openai_api_key, pinecone_api_key

import streamlit as st
import pandas as pd

from langchain_openai  import ChatOpenAI
from langchain_experimental.agents import create_pandas_dataframe_agent
from dotenv import load_dotenv, find_dotenv
from langchain_core.tools import tool
from sklearn.linear_model import LinearRegression

from langchain.schema import HumanMessage
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.prompts import (
                                SystemMessagePromptTemplate,
                                HumanMessagePromptTemplate,
                                ChatPromptTemplate,
                                MessagesPlaceholder
)
from streamlit_chat import message
from auxiliary_functions import *

#OpenAIKey
os.environ['OPENAI_API_KEY'] = openai_api_key
load_dotenv(find_dotenv())

#Title
st.title('AI Assistant for Data Science 🤖')

#Welcoming message
st.write("Hello, 👋 I am your AI Assistant and I am here to help you with your data science projects.")

#Explanation sidebar
with st.sidebar:
    st.write('*Your Data Science Adventure Begins with an CSV File.*')
    st.caption('''**You may already know that every exciting data science journey starts with a dataset.
    That's why I'd love for you to upload a CSV file.
    Once we have your data in hand, we'll dive into understanding it and have some fun exploring it.
    Then, we'll work together to shape your business challenge into a data science framework.
    I'll introduce you to the coolest machine learning models, and we'll use them to tackle your problem. Sounds fun right?**
    ''')

    st.divider()

    st.caption("<p style ='text-align:center'> made with ❤️ by Shiva</p>",unsafe_allow_html=True )

#Initialise the key in session state
def reset_app() -> None:
    """
    Clear any cached data and toggle a dummy session-state key so Streamlit
    fully re-runs the script whenever the user uploads a *different* file.
    """
    st.cache_data.clear()
    st.session_state.uploaded_file = None
    st.session_state["reset_trigger"] = not st.session_state.get("reset_trigger", False)

# ── Session-state initialisation ───────────────────────────────────────
if "reset_trigger" not in st.session_state:
    st.session_state["reset_trigger"] = False

if "uploaded_file" not in st.session_state:
    st.session_state.uploaded_file = None

if "clicked" not in st.session_state:
    # keeps track of which “Let’s get started” buttons have been pressed
    st.session_state.clicked = {1: False}

# ── Callback helper ────────────────────────────────────────────────────
def clicked(button_id: int) -> None:
    st.session_state.clicked[button_id] = True

# ── UI ─────────────────────────────────────────────────────────────────
st.button("Let's get started", on_click=clicked, args=[1])

if st.session_state.clicked[1]:
    tab1, tab2 = st.tabs(["Data Analysis and Data Science","ChatBox"])
    with tab1:
        user_csv = st.file_uploader("Upload your file here", type="csv")

        if user_csv is not None:
            if st.session_state.uploaded_file is None or st.session_state.uploaded_file != user_csv:
                reset_app()
                st.session_state.uploaded_file = user_csv

                # 🔄 Read the original data once (clean copy)
                user_csv.seek(0)
                df_original = pd.read_csv(user_csv, low_memory=False)
                st.session_state.df_original = df_original

                # 🧪 Read again for agent-specific usage (this one may be mutated)
                user_csv.seek(0)
                df_for_agent = pd.read_csv(user_csv, low_memory=False)
                st.session_state.df_for_agent = df_for_agent

            else:
                df_original = st.session_state.df_original
                df_for_agent = st.session_state.df_for_agent
        # <- reuse it safely
            st.write(list(df_original.columns))
            

            #llm model
            llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

            #Function sidebar
            @st.cache_data
            def steps_eda():
                steps_eda = llm([HumanMessage(content="What are the steps of EDA?")])
                return steps_eda
            
            @st.cache_data
            def data_science_framing():
                data_science_framing = llm([HumanMessage(content="Write a couple of paragraphs about the importance of framing a data science problem approriately")])
                return data_science_framing

            #Pandas agent
            pandas_agent = create_pandas_dataframe_agent(llm, df_for_agent, verbose = True, max_execution_time=120, allow_dangerous_code=True, handle_parsing_errors=True)
           
            #Functions main
            @st.cache_data
            def function_agent():
                st.write("**Data Overview**")
                st.write("The first rows of your dataset look like this:")
                st.write(df_original.head())
                st.write("**Data Cleaning**")
                columns_df = pandas_agent.run("What are the meaning of the columns?")
                st.write(columns_df)
                missing_values = pandas_agent.run("How many missing values does this dataframe have? Start the answer with 'There are'")
                st.write(missing_values)
                try:
                    duplicates = pandas_agent.run("Are there any duplicate values and if so where?")
                    st.write(duplicates)
                except ValueError as e:
                    st.warning("⚠️ Couldn't parse the agent response reliably. Falling back to pandas.")
                    duplicate_rows = df_original[df_original.duplicated()]
                    if duplicate_rows.empty:
                        st.write("✅ No duplicate rows found.")
                    else:
                        st.write(f"⚠️ Found {duplicate_rows.shape[0]} duplicate rows.")
                        st.write(duplicate_rows)
                st.write("**Data Summarisation**")
                st.write(df_original.describe())
                correlation_analysis = pandas_agent.run("Calculate correlations between numerical variables to identify potential relationships.")
                st.write(correlation_analysis)
                outliers = pandas_agent.run("Identify outliers in the data that may be erroneous or that may have a significant impact on the analysis.")
                st.write(outliers)
                new_features = pandas_agent.run("What new features would be interesting to create?.")
                st.write(new_features)
                st.write(df_original.describe())
                return

            def function_question_variable(user_question_variable):
                st.line_chart(df_original, y=[user_question_variable])
                summary_statistics = pandas_agent.run(f"Give me a summary of the statistics of {user_question_variable}")
                st.write(summary_statistics)
                normality = pandas_agent.run(f"Check for normality or specific distribution shapes of {user_question_variable}")
                st.write(normality)
                outliers = pandas_agent.run(f"Assess the presence of outliers of {user_question_variable}")
                st.write(outliers)
                trends = pandas_agent.run(f"Analyse trends, seasonality, and cyclic patterns of {user_question_variable}")
                st.write(trends)
                missing_values = pandas_agent.run(f"Determine the extent of missing values of {user_question_variable}")
                st.write(missing_values)

            
            @st.cache_data
            def function_question_dataframe():
                dataframe_info = pandas_agent.run(user_question_dataframe)
                st.write(dataframe_info)
                return

            

            #Main

            st.header('Exploratory data analysis')
            st.subheader('General information about the dataset')

            with st.sidebar:
                with st.expander('What are the steps of EDA'):
                    st.write(steps_eda())

            function_agent()

            st.subheader('Variable of study')
            user_question_variable = st.text_input('What variable are you interested in')
            if user_question_variable is not None and user_question_variable !="":
                function_question_variable(user_question_variable)

                st.subheader('Further study')

            if user_question_variable:
                user_question_dataframe = st.text_input( "Is there anything else you would like to know about your dataframe?")
                if user_question_dataframe is not None and user_question_dataframe not in ("","no","No"):
                    function_question_dataframe()
                if user_question_dataframe in ("no", "No"):
                    st.write("")
                
                if user_question_dataframe:
                    st.divider()
                    st.header("Data Science Problem")
                    st.write("Now that we have a solid grasp of the data at hand and a clear understanding of the variable we intend to investigate, it's important that we reframe our business problem into a data science problem.")

                    with st.sidebar:
                        with st.expander("The importance of framing a data science problem approriately"):
                            st.caption(data_science_framing())
                    
                    prompt = st.text_area('What is the business problem you would like to solve?')

                    if prompt:

                                           
                        def predict_stock_volume(input_value: str, input_variable: str, target_variable: str, df: pd.DataFrame) -> float:
                            """
                            Trains a linear regression model using the given dataframe, input variable, and target variable,
                            then predicts the target value for the provided input value.
        
                            Example use: Predict the volume on a specific date.
                            """
                            # --- Your logic, slightly adjusted for use inside tool ---
                            #st.write("📂 Columns in DataFrame:", list(df.columns))
                            #st.write("🔍 Input Variable:", input_variable)
                            #st.write("🎯 Target Variable:", target_variable)
                            col_map = {col.strip().lower(): col for col in df.columns}
                            target_col = col_map.get(target_variable.strip().lower())
                            #st.write("📊 Target Column:", target_col)
                            input_col = col_map.get(input_variable.strip().lower())
                            #st.write("📈 Input Column:", input_col)

                            if not target_col or not input_col:
                                raise ValueError("One or both of the specified columns do not exist in the dataframe.")

                            if df[input_col].dtype == 'object':
                                try:
                                    df[input_col] = pd.to_datetime(df[input_col])
                                except:
                                    raise ValueError("Input variable could not be converted to datetime.")

                            if pd.api.types.is_datetime64_any_dtype(df[input_col]):
                                df['__input_num__'] = df[input_col].map(pd.Timestamp.toordinal)
                                input_value = input_value.split("Values:")[-1].strip()
                                input_num = pd.to_datetime(input_value).toordinal()
                            else:
                                df['__input_num__'] = df[input_col]
                                input_num = float(input_value)

                            df_clean = df[['__input_num__', target_col]].dropna()
                            model = LinearRegression()
                            model.fit(df_clean[['__input_num__']], df_clean[target_col])

                            predicted_value = model.predict([[input_num]])
                            return float(predicted_value[0])

                    # Defining the system prompt (how the AI should act)
                    system_prompt = SystemMessagePromptTemplate.from_template("""You are an AI assistant that converts complex business problems into data science problems to solve using linear regression.
                                                                            Context: {{context}}
                                                                            """)

                    # the user prompt is provided by the user, in this case however the only dynamic input is the business_problem
                    user_prompt = HumanMessagePromptTemplate.from_template("""Convert the following business problem into a data science problem to solve using linear regression: {business_problem}. Identify the input variable, target variable, input value separately. Analyse the provided context to identify the target and input variables. Use only the column names in the context to identify the target and input variables. Do not use any other variables and print the names as it is in the context.
                                                                        Also extract the values of input variable from the {business_problem}. Do not add any content just the values""",
                                                                        input_variables=["business_problem"]
                                                                        )
                    
                    first_prompt = ChatPromptTemplate.from_messages([system_prompt, user_prompt])

                    chain_one = (
                                {"business_problem": lambda x: x["business_problem"]}
                                | first_prompt
                                | llm
                                | {"business_problem_sol": lambda x: x}
                                )
                    msg = chain_one.invoke({"business_problem": prompt, "context":df_original.to_string(index=False)})
                    #st.write(msg)

                
                    content_str = msg["business_problem_sol"].content 

                    # Step 2: Split and clean lines
                    lines = [line.strip() for line in content_str.strip().split('\n') if line.strip()]

                    # Step 3: Initialize variables
                    target_variable = input_variable = input_value = None

                    # Step 4: Extract variables from lines
                    for line in lines:
                        if line.lower().startswith("target variable:"):
                            target_variable = line.split(":", 1)[1].strip()
                        elif line.lower().startswith("input variable:"):
                            input_variable = line.split(":", 1)[1].strip()
                        elif line.lower().startswith("input value:"):
                            input_value = line.split(":", 1)[1].strip().strip("'").strip('"')

                    # Step 5: Display or use in Streamlit
                    st.write("📊 Target Variable:", target_variable)
                    st.write("📈 Input Variable:", input_variable)
                    st.write("🔢 Input Value:", input_value)
                    #st.write(list(df_original.columns))

                    result = predict_stock_volume(input_value = input_value,
                                                        input_variable = input_variable,
                                                        target_variable = target_variable,
                                                        df = df_original
                                                        )
                    st.write(result)

    with tab2:
        st.header("ChatBox")
        st.write("🤖 Welcome to the AI Assistant ChatBox!") 
        st.write("Got burning questions about your data science problem or need help navigating the intricacies of your project? You're in the right place! Our chatbot is geared up to assist you with insights, advice, and solutions. Just type in your queries, and let's unravel the mysteries of your data together! 🔍💻")

        st.write("")
        if 'responses' not in st.session_state:
            st.session_state['responses'] = ["How can I assist you?"]
        if 'requests' not in st.session_state:
            st.session_state['requests'] = []

        llm = ChatOpenAI(model_name="gpt-3.5-turbo", openai_api_key=openai_api_key)

        if 'buffer_memory' not in st.session_state:
            st.session_state.buffer_memory=ConversationBufferWindowMemory(k=3,return_messages=True)


        system_msg_template = SystemMessagePromptTemplate.from_template(template="""Answer the question as truthfully as possible using the provided context, 
        and if the answer is not contained within the text below, say 'I don't know'""")
        human_msg_template = HumanMessagePromptTemplate.from_template(template="{input}")
        prompt_template = ChatPromptTemplate.from_messages([system_msg_template, MessagesPlaceholder(variable_name="history"), human_msg_template])

        conversation = ConversationChain(memory=st.session_state.buffer_memory, prompt=prompt_template, llm=llm, verbose=True)

        response_container = st.container()
        textcontainer = st.container()

       
        with textcontainer:
            query = st.text_input("Hello! How can I help you? ", key="input")
            if query:
                with st.spinner("thinking..."):
                    conversation_string = get_conversation_string()
                    refined_query = query_refiner(conversation_string, query)
                    st.subheader("Refined Query:")
                    st.write(refined_query)
                    context = find_match(refined_query)
                    response = conversation.predict(input=f"Context:\n {context} \n\n Query:\n{query}")
                st.session_state.requests.append(query)
                st.session_state.responses.append(response)

        with response_container:
            if st.session_state['responses']:
                for i in range(len(st.session_state['responses'])):
                    message(st.session_state['responses'][i],key=str(i))
                    if i < len(st.session_state['requests']):
                        message(st.session_state["requests"][i], is_user=True,key=str(i)+ '_user')
                
              

                    
    