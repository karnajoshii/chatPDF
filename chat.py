import streamlit as st
import pandas as pd
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.embeddings.spacy_embeddings import SpacyEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.tools.retriever import create_retriever_tool
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI  
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnablePassthrough
from langchain_community.utilities import SQLDatabase
from langchain_core.output_parsers import StrOutputParser

# Load environment variables from .env file
load_dotenv()

# Load SQL Database credentials from environment variables
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "3306")
DB_USER = os.getenv("DB_USER", "root")
DB_PASSWORD = os.getenv("DB_PASSWORD", "Karna!21")
DB_NAME = os.getenv("DB_NAME", "AI")

# Use OpenAI embeddings
embeddings = SpacyEmbeddings(model_name="en_core_web_sm")

# Function to read text from PDFs
def pdf_read(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text() or ""
    return text

# Function to read text from CSVs
def csv_read(csv_docs):
    text = ""
    for csv in csv_docs:
        df = pd.read_csv(csv)
        text += "\n".join(df.astype(str).apply(lambda x: ', '.join(x), axis=1))
    return text

# Function to split text into chunks
def get_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=200)
    return text_splitter.split_text(text)

# Function to store chunks in FAISS vector store
def vector_store(text_chunks):
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_db")

# Function to check if a query is order-related
def is_order_related(query):
    """Use LLM to classify if the query is about order details."""
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=os.getenv("OPENAI_API_KEY"))
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """Determine if the following user query is related to order details. 
        Do not consider queries about order modifications or general order placement as order-related.
        Return 'YES' for order details queries and 'NO' otherwise."""),
        ("human", "{query}")
    ])
    
    chain = prompt | llm | StrOutputParser()
    
    try:
        response = chain.invoke({"query": query}).strip().upper()
        return response == "YES"
    except Exception as e:
        print(f"LLM order detection failed: {e}")
        return False

# Function to initialize SQL Database
def init_database():
    """Initialize the database connection using environment variables."""
    try:
        user = DB_USER
        password = DB_PASSWORD
        host = DB_HOST
        port = DB_PORT
        database = DB_NAME
        if not all([user, password, host, port, database]):
            raise ValueError("Missing one or more database environment variables.")
        port = int(port)
        db_uri = f"mysql+mysqlconnector://{user}:{password}@{host}:{port}/{database}"
        return SQLDatabase.from_uri(db_uri)
    except Exception as e:
        raise RuntimeError(f"Database connection failed: {str(e)}")

# Function to detect specific entities in the query
def extract_query_details(query):
    """Use LLM to determine if the query contains specific details like names or IDs."""
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=os.getenv("OPENAI_API_KEY"))
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """Analyze the following user query and determine if it contains specific details such as a customer name, order ID, or other identifiable value.
        Return a JSON object with:
        - 'has_details': true/false (indicating if specific details are present)
        - 'details': the extracted value (e.g., 'John Doe', '12345') or null if none found"""),
        ("human", "{query}")
    ])
    
    chain = prompt | llm | StrOutputParser()
    
    try:
        import json
        response = chain.invoke({"query": query})
        return json.loads(response)
    except Exception as e:
        print(f"Entity extraction failed: {e}")
        return {"has_details": False, "details": None}

# Function to generate responses from FAISS (document-based)
def get_conversational_chain(tools, ques):
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=os.getenv("OPENAI_API_KEY"), verbose=True)

    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a helpful assistant. Use past conversation to understand user intent. 
                      If the user asks a follow-up question, refer to previous messages for context.
                      If you get any information related to {input} than pass that and if you are not confident than say that this is the information if you want more specific data regarding {input} you contact the suppport.
                      CRITICAL : No answers must be out of the documents.I want nearest answers if you are not sure.
                      If getting the same thing for multiple products don't just randomly give the any product details ask the user for which prpoduct you need the information eg. user says Give me the box contents now there may be the box contents for several products so ask the user that please specify the product you what the box contents detail for.
                      If the answer is not in the documents, say: 'Sorry, I couldn't find relevant information.'
                      """),
        ("placeholder", "{chat_history}"),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ])

    tool = [tools]
    agent = create_tool_calling_agent(llm, tool, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tool, verbose=True)

    response = agent_executor.invoke({
        "input": ques,
        "chat_history": st.session_state.chat_history  
    })

    return response['output']

# Function to create SQL chain
def get_sql_chain(db):
    template = """
        You are a data analyst at a company. You are interacting with a user who is asking you questions about the company's database.
        Based on the table schema below, write a SQL query that would answer the user's question. Take the conversation history into account.
        Make sure never query whole database like "SELECT * from orders" Be specific.
        Additionally, handle common greetings and small talk:
        - If the user says "hi", "hello", "hey", respond with a friendly greeting.
        - If they say "thank you", reply with "You're welcome!".
        - If they say "bye", "goodbye", "see you", wish them a great day.
        Not only this but if any greetings respond it correctly in friendly manner.

        CRITICAL : Don't include <SCHEMA>{schema}</SCHEMA> in response

        Conversation History: {chat_history}

        Write only the SQL query and nothing else. Do not wrap the SQL query in any other text, not even backticks.

        Your turn:
        Question: {question}
        SQL Query:
        """
    
    prompt = ChatPromptTemplate.from_template(template)
    llm = ChatOpenAI(model="gpt-4-0125-preview")

    def get_schema(_):
        return db.get_table_info()
    
    return (
        RunnablePassthrough.assign(schema=get_schema)
        | prompt
        | llm
        | StrOutputParser()
    )

# Function to get response from MySQL for order-related queries
def get_response(user_query: str, db: SQLDatabase, chat_history: list):
    sql_chain = get_sql_chain(db)

    template = """
        You are a data analyst at a company. You are interacting with a user who is asking you questions about the company's database.
        Based on the table schema below, question, sql query, and sql response, write a natural language response.
        Additionally, handle common greetings and small talk:
        - If the user says "hi", "hello", "hey", respond with a friendly greeting.
        - If they say "thank you", reply with "You're welcome!".
        - If they say "bye", "goodbye", "see you", wish them a great day.
        Not only this but if any greetings respond it correctly in friendly manner.

        CRITICAL : Don't include <SCHEMA>{schema}</SCHEMA> in response

        Conversation History: {chat_history}
        SQL Query: <SQL>{query}</SQL>
        User question: {question}
        SQL Response: {response}
    """

    prompt = ChatPromptTemplate.from_template(template)
    llm = ChatOpenAI(model="gpt-4-0125-preview")

    try:
        # Extract details from the query
        query_details = extract_query_details(user_query)

        # Check if query contains specific details
        if not query_details["has_details"]:
            # If no specific details provided, move to pending state
            if "pending_query" not in st.session_state:
                st.session_state.pending_query = user_query
                return "I need more information to provide the details. Please specify the customer name, order ID, or another relevant detail."
            else:
                # If there's a pending query, assume this is the follow-up input
                combined_query = f"{st.session_state.pending_query} for {user_query}"
                del st.session_state.pending_query  # Clear pending query
                user_query = combined_query

        chain = (
            RunnablePassthrough.assign(query=sql_chain).assign(
                schema=lambda _: db.get_table_info(),
                response=lambda vars: db.run(vars["query"]),  
            )
            | prompt
            | llm
            | StrOutputParser()
        )

        response = chain.invoke({
            "question": user_query,
            "chat_history": chat_history,
        })
        return response
    
    except Exception as e:
        return f"SQL Error: {str(e)}. Please refine your query."

# Function to handle user input and get response
def user_input(user_question):
    # If the query is related to orders, query MySQL
    if is_order_related(user_question):
        try:
            if "db" not in st.session_state:
                st.session_state.db = init_database()

            response = get_response(user_question, st.session_state.db, st.session_state.chat_history)

            if "SQL Error" in response or "Error" in response:
                return response

            # Handle pending query prompt separately
            if "I need more information" in response:
                st.session_state.chat_history.append(HumanMessage(content=user_question))
                st.session_state.chat_history.append(AIMessage(content=response))
                return response

            st.session_state.chat_history.append(HumanMessage(content=user_question))
            st.session_state.chat_history.append(AIMessage(content=response))
            return response

        except Exception as e:
            return f"Database error: {str(e)}. Please check MySQL connection."

    # Otherwise, use FAISS for document-based queries
    try:
        new_db = FAISS.load_local("faiss_db", embeddings, allow_dangerous_deserialization=True)
        retriever = new_db.as_retriever()
        retrieval_chain = create_retriever_tool(retriever, "data_extractor", "This tool answers queries from uploaded documents.")
        response = get_conversational_chain(retrieval_chain, user_question)

        st.session_state.chat_history.append(HumanMessage(content=user_question))
        st.session_state.chat_history.append(AIMessage(content=response))
        return response

    except Exception as e:
        return f"Document Retrieval Error: {str(e)}"

# Main function for Streamlit interface
def main():
    st.set_page_config(page_title="Chat with Documents & Orders", layout="centered")
    st.header("Chat with Documents & Orders")

    # Initialize chat history
    if 'qa_pairs' not in st.session_state:
        st.session_state.qa_pairs = []
    
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [AIMessage(content="Hello! Ask me anything.")]

    # Display chat history
    for speaker, question, answer in st.session_state.qa_pairs:
        if speaker == "user":
            st.markdown(
                f'<div style="background-color:#DCF8C6;padding:10px;border-radius:10px;margin-bottom:10px;">You: {question}</div>', 
                unsafe_allow_html=True
            )
        st.markdown(
            f'<div style="background-color:#F1F0F0;padding:10px;border-radius:10px;margin-bottom:20px;">Assistant: {answer}</div>', 
            unsafe_allow_html=True
        )

    with st.form(key="chat_form"):
        user_question = st.text_input("Type your message here...")
        submit_button = st.form_submit_button(label="Send")

    if submit_button and user_question:
        answer = user_input(user_question)
        st.session_state.qa_pairs.append(("user", user_question, answer))
        st.rerun()

    with st.sidebar:
        st.title("Upload Documents")
        pdf_docs = st.file_uploader("Upload PDFs", accept_multiple_files=True, type=["pdf"])
        csv_docs = st.file_uploader("Upload CSVs", accept_multiple_files=True, type=["csv"])

        if st.button("Process"):
            with st.spinner("Processing..."):
                raw_text = pdf_read(pdf_docs) + csv_read(csv_docs)
                text_chunks = get_chunks(raw_text)
                vector_store(text_chunks)
                st.success("Processing complete!")

if __name__ == "__main__":
    main()