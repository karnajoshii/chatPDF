from flask import Flask, request, jsonify
import pandas as pd
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import OpenAIEmbeddings  # Switch to OpenAI embeddings
from langchain_community.vectorstores import FAISS
from langchain.tools.retriever import create_retriever_tool
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI  
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnablePassthrough
from langchain_community.utilities import SQLDatabase
from langchain_core.output_parsers import StrOutputParser
import tempfile
import json
from flask_swagger_ui import get_swaggerui_blueprint
from flask_cors import CORS
import jwt as pyjwt
import datetime
import mysql.connector
from mysql.connector import Error
from werkzeug.security import generate_password_hash, check_password_hash
import re

# Swagger UI configuration
SWAGGER_URL = '/swagger'  
API_URL = '/static/swagger.yaml'


# Create Swagger UI blueprint
swaggerui_blueprint = get_swaggerui_blueprint(
    SWAGGER_URL,
    API_URL,
    config={
        'app_name': "Chatbot API"
    }
)

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)
# Register the blueprint with the Flask app
app.register_blueprint(swaggerui_blueprint, url_prefix=SWAGGER_URL)

CORS(app)

# Load SQL Database credentials from environment variables
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "3306")
DB_USER = os.getenv("DB_USER", "root")
DB_PASSWORD = os.getenv("DB_PASSWORD", "Karna!21")
DB_NAME = os.getenv("DB_NAME", "AI")
JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY","651084d929f5ebe56ffb6057c38e19feb8bfbac5615a5f815e05dd2958ae9eb3c1f24a62aca670d96ce0d6c2a00e7160d9be98c5bf2142a988ac99458713c4f5a0ded258e3d31704aae98c53825fcc19b3d9ec674cbb6baead16d28da8b77e406edfa136b7f28b31e4ee0fea6bc71b75f190a2c722e1914459efec21a1796c9d")
JWT_EXPIRATION_DELTA = datetime.timedelta(days=1)  # Token valid for 1 day


# Use OpenAI embeddings for better semantic search
embeddings = OpenAIEmbeddings(model="text-embedding-ada-002", api_key=os.getenv("OPENAI_API_KEY"))

# Global variables
chat_history = []
db = None
pending_query = None


# Function to read text from PDFs
def pdf_read(pdf_file_path):
    text = ""
    pdf_reader = PdfReader(pdf_file_path)
    for page in pdf_reader.pages:
        text += page.extract_text() or ""
    return text

# Function to read text from CSVs
def csv_read(csv_file_path):
    text = ""
    df = pd.read_csv(csv_file_path)
    text += "\n".join(df.astype(str).apply(lambda x: ', '.join(x), axis=1))
    return text

# Function to split text into chunks - increased chunk size for better context
def get_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return text_splitter.split_text(text)

# Function to store chunks in FAISS vector store
def vector_store(text_chunks):
    # Add source metadata to chunks for better traceability
    metadatas = [{"source": f"chunk_{i}", "content": chunk[:100]} for i, chunk in enumerate(text_chunks)]
    
    vector_store = FAISS.from_texts(
        text_chunks, 
        embedding=embeddings,
        metadatas=metadatas
    )
    vector_store.save_local("faiss_openai_db")
    return True

# Function to check if a query is order-related
def is_order_related(query):
    """Use LLM to classify if the query is about order details."""
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=os.getenv("OPENAI_API_KEY"))
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """Determine if the following user query is related to order details. 
        Do not consider queries about order modifications or general order placement as order-related.
        CRITICAL: Be specific to the order related. 
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

# Function to extract query details
def extract_query_details(query):
    print(query)
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=os.getenv("OPENAI_API_KEY"))
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """Analyze the following user query and determine if it contains specific details such as a customer name, order ID, or other identifiable value.
        Consider that:
        - An ID is typically a number (e.g., '12345') or alphanumeric code (e.g., 'ORD-123').
        - Product names (e.g., 'quantum 400') or random questions should NOT be considered customer names or IDs.
        
        Return a JSON object with:
        - 'has_details': true/false (indicating if a ID is present)
        - 'details': the extracted value (e.g., 'Luna', '12345') or null if none found
        - 'type': 'name', 'id', or null (indicating the type of detail, if any)"""),
        ("human", "{query}")
    ])
    
    chain = prompt | llm | StrOutputParser()
    
    try:
        response = chain.invoke({"query": query})
        return json.loads(response)
    except Exception as e:
        print(f"Entity extraction failed: {e}")
        return {"has_details": False, "details": None, "type": None}

# Function to check if the user doesn't know the requested information
def is_lack_of_knowledge_response(response):
    """Use OpenAI to determine if the user's response indicates they don't know the requested information."""
    try:
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=os.getenv("OPENAI_API_KEY"))
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", """Determine if the following user message indicates they DO NOT KNOW or DO NOT HAVE the information 
            that was previously requested from them (such as a customer ID, name, or other specific detail).
            
            Examples that indicate lack of knowledge:
            - "I don't know"
            - "not sure about that"
            - "don't have that information"
            - "can you help me find it"
            - "I have no idea"
            - "I forgot"
            
            Return ONLY 'YES' if the message indicates they don't know the information.
            Return ONLY 'NO' if the message appears to provide information or asks a completely different question."""),
            ("human", "{response}")
        ])
        
        chain = prompt | llm | StrOutputParser()
        
        result = chain.invoke({"response": response}).strip().upper()
        return result == "YES"
    except Exception as e:
        print(f"LLM knowledge detection failed: {e}")
        # Fallback to basic pattern matching if API call fails
        lack_of_knowledge_phrases = [
            "i don't know", "don't know", "no idea", "not sure", 
            "i have no idea", "can't tell", "don't have that", 
            "not available", "i'm not sure", "dunno", "no clue"
        ]
        
        response_lower = response.lower().strip()
        for phrase in lack_of_knowledge_phrases:
            if phrase in response_lower:
                return True
        return False
    
faiss_index = FAISS.load_local("faiss_openai_db", embeddings,allow_dangerous_deserialization=True)
def retrieve_documents(query, k=5):
    """Retrieves top-k most relevant documents from FAISS."""
    docs = faiss_index.similarity_search(query, k=k)
    return "\n".join([doc.page_content for doc in docs]) if docs else "No relevant data found."

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=os.getenv("OPENAI_API_KEY"), verbose=True)

# Function to generate responses from FAISS (document-based)
def get_conversational_chain(tools,ques):
    """Handles user queries using RAG (retrieval + generation)."""

    # Step 1: Retrieve documents using FAISS
    retrieved_data = retrieve_documents(ques)
    print(f"User Query: {ques}")
    print(f"Retrieved Chunks: {retrieved_data}")

    # If no relevant data found, return a fallback response
    if not retrieved_data:
        return "I only provide details about Feniex Quantum products. Unfortunately, I don’t have information on that topic."

    # Step 2: Define system prompt
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a helpful Feniex AI assistant. Answer user queries based ONLY on the retrieved documents.
        
        CRITICAL RULES:
        1. Always retrieve documents before answering.
        2. Never generate answers from general knowledge—only use retrieved data.
        3. If multiple products are mentioned, ask the user to specify.
        4. If the user asks about "Quantum 2.0 27\" Mini", DO NOT provide details.
        5. CRITICAL: Your answers must be based solely on the retrieved documents. Never make up information.
        6. IMPORTANT : If the user is not specific about the product detail and the retrieved data contains multiple products, ask the user to specify which product they need information about instead of providing all data.
        7. IMPORTANT : Only provide the details of this quantum products 1.Quantum 400 2. Quantum 600 3. Quantum 800 4. Quantum Arrow Board 5. Quantum Pillar light 6 . ROCKER PANEL 7. Quantum split arrow board
        Don't provide any details of Quantum 2.0 27" Mini
        You are expert in giving product details of Feneix products.That are your strenght or can say capabilities
        8. REMEMBER : You are feneix AI Assistance for product details.You can reply this if someone ask about you or your name anyting related to you.
        9. REMEMBER : If something is ask about contacting sales team than say call : 800.615.8350 email : support@feniex.com
        10.CRITICAL : If you don't find answer than say that I’m here to assist with Feniex Quantum 2.0 products. Unfortunately, I don’t have details about [specific topic] in this context. How can I help you with our products today?
        11.CRITICAL : Never say that I don't have information in documents NEVER include that you are trained on the documents say I don’t have details about [specific topic] in this context. How can I help you with our products today?
        12.CRITICAL : If the user is not specific about the product detail and the retrieved data contains multiple products, ask the user to specify which product they need information about instead of providing all data.
        13.CRITICAL : Setup and installation both are same VEry IMPORTANT.
         Use the retrieved documents and past conversation to understand user intent. 
                      If the user asks a follow-up question, refer to previous messages for context.
                      CRITICAL: Don't use your own knowledge only based on retrived documents.Simply return that the question is out of context.
                      IMPORTANT : Don't use your own knowledge use the data retrived from Vector database only.If data not found simply return the question is out of context.
                      
                      IMPORTANT : If user is not specific about the product detail ask the product detail first if the answer contains several products don't just pass the whole data.
                      When you get information related to the user's query:
                      1. Provide the most accurate answer based on the retrieved documents
                      2. If you're not confident, indicate that the information is based on the documents available
                      3. Cite the source of information when possible
                      
                      CRITICAL: Your answers must be based solely on the retrieved documents. Never make up information.
                      
                      If multiple products are mentioned in the documents, don't randomly choose one. Ask the user to 
                      specify which product they need information about.
                      
                      If the answer is not in the documents, say: 'Sorry, I couldn't find relevant information about [specific topic].
                      CRITICAL : Don't include that I can't find in provided documents.
                      CRITICAL : If the question is out of context than don't ask any question like be specific or so just say "I’m here to assist with Feniex quantum 2.0 products . Unfortunately, I don’t have details about [specific topic] in this context. How can I help you with our products today?
        Retrieved Data:
        {context}
        
        User Query: {query}
        Response:"""),
    ])

    # Step 3: Generate response
    final_prompt = prompt.format(context=retrieved_data, query=ques)
    response = llm.predict(final_prompt)
    print(f"Generated Response: {response}")
    return response

# Function to create SQL chain
def get_sql_chain(db_conn):
    template = """
        You are a data analyst at a company. You are interacting with a user who is asking you questions about the company's database.
        Based on the table schema below, write a SQL query that would answer the user's question. Take the conversation history into account.
        Make sure to never query the whole database with queries like "SELECT * from orders". Be specific and limit your results.
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
        return db_conn.get_table_info()
    
    return (
        RunnablePassthrough.assign(schema=get_schema)
        | prompt
        | llm
        | StrOutputParser()
    )

# Function to get response from MySQL for order-related queries
def get_response(user_query, db_conn, chat_hist):
    global pending_query
    
    sql_chain = get_sql_chain(db_conn)

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
            if pending_query is None:
                pending_query = user_query
                return "I need more information to provide the details. Please specify the order ID."
            else:
                # If there's a pending query, assume this is the follow-up input
                combined_query = f"{pending_query} for {user_query}"
                pending_query = None  # Clear pending query
                user_query = combined_query

        chain = (
            RunnablePassthrough.assign(query=sql_chain).assign(
                schema=lambda _: db_conn.get_table_info(),
                response=lambda vars: db_conn.run(vars["query"]),  
            )
            | prompt
            | llm
            | StrOutputParser()
        )

        response = chain.invoke({
            "question": user_query,
            "chat_history": chat_hist,
        })
        
        return response
    
    except Exception as e:
        return f"SQL Error: {str(e)}. Please refine your query."

# Function to handle user input and get response
def process_user_input(user_question):
    global chat_history, db, pending_query
    
    # Check if there's a pending query from a previous order-related question
    if pending_query is not None:
        # Extract details from the new input
        query_details = extract_query_details(user_question)
        
        # If the input contains a name or ID, combine with pending query
        if query_details["has_details"] and query_details["type"] in ["name", "id"]:
            combined_query = f"{pending_query} for {user_question}"
            temp_pending = pending_query  # Store temporarily for logging
            pending_query = None  # Clear pending query
            
            try:
                if db is None:
                    db = init_database()

                response = get_response(combined_query, db, chat_history)

                if "SQL Error" in response or "Error" in response:
                    return {"status": "error", "message": response}

                chat_history.append(HumanMessage(content=combined_query))
                chat_history.append(AIMessage(content=response))
                return {"status": "success", "message": response, "pending_processed": True, "original_query": temp_pending}

            except Exception as e:
                return {"status": "error", "message": f"Database error: {str(e)}. Please check MySQL connection."}
        
        # If it's a random question (no name or ID), complete the pending query and switch context
        else:
            # Complete the pending query with a default response
            pending_response = "I can’t provide order details order ID. Let me help with your new question."
            chat_history.append(HumanMessage(content=pending_query))
            chat_history.append(AIMessage(content=pending_response))
            pending_query = None  # Clear pending query
            
            # Treat the new input as a fresh query
            try:
                new_db = FAISS.load_local("faiss_openai_db", embeddings, allow_dangerous_deserialization=True)
                retriever = new_db.as_retriever(search_kwargs={"k": 5})
                retrieval_chain = create_retriever_tool(
                    retriever, 
                    "data_extractor", 
                    "This tool searches through uploaded documents to answer queries. It works best with specific questions."
                )
                response = get_conversational_chain(retrieval_chain, user_question)

                chat_history.append(HumanMessage(content=user_question))
                chat_history.append(AIMessage(content=response))
                return {"status": "success", "message": response, "pending_cleared": True}

            except Exception as e:
                return {"status": "error", "message": f"Document Retrieval Error: {str(e)}"}
    
    # If no pending query, evaluate the new question
    if is_order_related(user_question):
        try:
            if db is None:
                db = init_database()

            # Extract details from the query
            query_details = extract_query_details(user_question)
            print(f"QUERY : {query_details}")

            # If specific details are provided (e.g., "Luna" or "I want information about Luna")
            if query_details["has_details"]:
                response = get_response(user_question, db, chat_history)

                if "SQL Error" in response or "Error" in response:
                    return {"status": "error", "message": response}

                chat_history.append(HumanMessage(content=user_question))
                chat_history.append(AIMessage(content=response))
                return {"status": "success", "message": response}
            else:
                # If no details, set pending query and ask for clarification
                pending_query = user_question
                response = "I need more information to provide the details. Please specify the order ID."
                chat_history.append(HumanMessage(content=user_question))
                chat_history.append(AIMessage(content=response))
                return {"status": "pending", "message": response, "pending_query": pending_query}

        except Exception as e:
            return {"status": "error", "message": f"Database error: {str(e)}. Please check MySQL connection."}

    # Otherwise, use FAISS for document-based queries
    try:
        new_db = FAISS.load_local("faiss_openai_db", embeddings, allow_dangerous_deserialization=True)
        retriever = new_db.as_retriever(search_kwargs={"k": 5})
        retrieval_chain = create_retriever_tool(
            retriever, 
            "data_extractor", 
            "This tool searches through uploaded documents to answer queries. It works best with specific questions."
        )
        response = get_conversational_chain(retrieval_chain, user_question)

        chat_history.append(HumanMessage(content=user_question))
        chat_history.append(AIMessage(content=response))
        return {"status": "success", "message": response}

    except Exception as e:
        return {"status": "error", "message": f"Document Retrieval Error: {str(e)}"}
# API Routes

@app.route('/api/query', methods=['POST'])
def query_endpoint():
    try:
        data = request.json
        if not data or 'query' not in data:
            return jsonify({"status": "error", "message": "Query parameter is required"}), 400
        
        user_question = data['query']
        result = process_user_input(user_question)
        return jsonify(result)
    
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/api/upload-document', methods=['POST'])
def upload_document():
    try:
        if 'file' not in request.files:
            return jsonify({"status": "error", "message": "No file part"}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({"status": "error", "message": "No file selected"}), 400
        
        # Create temporary file
        temp_file = tempfile.NamedTemporaryFile(delete=False)
        file.save(temp_file.name)
        
        # Process the file based on its type
        file_extension = file.filename.split('.')[-1].lower()
        
        if file_extension == 'pdf':
            raw_text = pdf_read(temp_file.name)
        elif file_extension == 'csv':
            raw_text = csv_read(temp_file.name)
        else:
            os.unlink(temp_file.name)
            return jsonify({"status": "error", "message": "Unsupported file type. Please upload PDF or CSV files."}), 400
        
        # Process text
        text_chunks = get_chunks(raw_text)
        vector_store(text_chunks)
        
        # Clean up temporary file
        os.unlink(temp_file.name)
        
        return jsonify({
            "status": "success", 
            "message": f"Document '{file.filename}' processed successfully",
            "chunks_created": len(text_chunks)
        })
    
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/api/batch-upload', methods=['POST'])
def batch_upload():
    try:
        if 'files[]' not in request.files:
            return jsonify({"status": "error", "message": "No files part"}), 400
        
        files = request.files.getlist('files[]')
        if not files or files[0].filename == '':
            return jsonify({"status": "error", "message": "No files selected"}), 400
        
        all_text = ""
        processed_files = []
        
        for file in files:
            # Create temporary file
            temp_file = tempfile.NamedTemporaryFile(delete=False)
            file.save(temp_file.name)
            
            # Process the file based on its type
            file_extension = file.filename.split('.')[-1].lower()
            
            if file_extension == 'pdf':
                file_text = pdf_read(temp_file.name)
            elif file_extension == 'csv':
                file_text = csv_read(temp_file.name)
            else:
                os.unlink(temp_file.name)
                continue  # Skip unsupported files
            
            all_text += file_text
            processed_files.append(file.filename)
            
            # Clean up temporary file
            os.unlink(temp_file.name)
        
        if not all_text:
            return jsonify({"status": "error", "message": "No supported files were processed"}), 400
        
        # Process all text together
        text_chunks = get_chunks(all_text)
        vector_store(text_chunks)
        
        return jsonify({
            "status": "success", 
            "message": f"Processed {len(processed_files)} documents successfully",
            "processed_files": processed_files,
            "chunks_created": len(text_chunks)
        })
    
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/api/chat-history', methods=['GET'])
def get_chat_history():
    global chat_history
    formatted_history = []
    
    for message in chat_history:
        if isinstance(message, HumanMessage):
            formatted_history.append({"role": "user", "content": message.content})
        elif isinstance(message, AIMessage):
            formatted_history.append({"role": "assistant", "content": message.content})
    
    return jsonify({
        "status": "success",
        "history": formatted_history,
        "pending_query": pending_query
    })

@app.route('/api/clear-history', methods=['POST'])
def clear_chat_history():
    global chat_history, pending_query
    
    chat_history = []
    pending_query = None
    
    return jsonify({"status": "success", "message": "Chat history cleared"})

@app.route('/api/db-status', methods=['GET'])
def check_db_status():
    global db
    
    try:
        if db is None:
            db = init_database()
        
        # Try a simple query to test the connection
        test_result = db.run("SELECT 1 as test")
        tables = db.get_usable_table_names()
        
        return jsonify({
            "status": "success", 
            "message": "Database connection is active",
            "tables": tables
        })
    except Exception as e:
        return jsonify({
            "status": "error", 
            "message": f"Database connection error: {str(e)}",
            "connection_string": f"{DB_HOST}:{DB_PORT}/{DB_NAME}"
        }), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    # Check if FAISS index exists
    faiss_db_exists = os.path.exists("faiss_openai_db")
    
    return jsonify({
        "status": "healthy",
        "api_version": "1.0.0",
        "document_db": "available" if faiss_db_exists else "not initialized",
        "pending_query": pending_query is not None,
        "chat_history_length": len(chat_history)
    })

def create_db_connection():
    try:
        connection = mysql.connector.connect(
            host=DB_HOST,
            port=DB_PORT,
            user=DB_USER,
            password=DB_PASSWORD,
            database=DB_NAME
        )
        return connection
    except Error as e:
        print(f"Database connection error: {e}")
        return None

# Function to execute SQL query
def execute_query(connection, query, params=None, fetch=True):
    cursor = connection.cursor(dictionary=True)  # Use dictionary cursor for readable results
    try:
        cursor.execute(query, params)
        if fetch:  # For SELECT queries
            result = cursor.fetchall()  # or fetchone() depending on your need
            connection.commit()  # Commit only if necessary (e.g., autocommit is off)
            return result
        else:  # For INSERT, UPDATE, DELETE
            connection.commit()
            return cursor  # Return the cursor to access lastrowid or other attributes
    except Error as e:
        print(f"Error executing query: {e}")
        connection.rollback()
        raise e
    finally:
        cursor.close()  # Safe to close here since results are returned



# Function to find user by email
def find_user_by_email(email):
    connection = create_db_connection()  # Assuming this function exists
    if connection is None:
        return None
    
    try:
        query = "SELECT * FROM users WHERE email = %s"
        results = execute_query(connection, query, (email,), fetch=True)
        # Since we're expecting one user, take the first result (if any)
        user = results[0] if results else None
        return user
    except Error as e:
        print(f"Error finding user: {e}")
        return None
    finally:
        if connection and connection.is_connected():
            connection.close()
            print("Database connection closed")

# Function to create JWT token - FIXED
def create_token(user_id):
    payload = {
        'exp': datetime.datetime.now(datetime.UTC) + JWT_EXPIRATION_DELTA,
        'iat': datetime.datetime.now(datetime.UTC),
        'sub': str(user_id)
    }
    # Use PyJWT's encode method and ensure we convert to string
    token = pyjwt.encode(payload, JWT_SECRET_KEY, algorithm='HS256')
    
    # Convert bytes to string if token is in bytes format (PyJWT 2.0+)
    if isinstance(token, bytes):
        return token.decode('utf-8')
    return token

def is_valid_email(email):
    """
    Validates if the given string is a valid email address.
    
    Args:
        email (str): The email address to validate
        
    Returns:
        bool: True if the email is valid, False otherwise
    """
    # Regular expression pattern for basic email validation
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    
    # Check if the email matches the pattern
    if re.match(pattern, email):
        return True
    else:
        return False

# Registration API
@app.route('/api/register', methods=['POST'])
def register():
    try:
        data = request.json
        
        # Validate request data
        if not data or 'email' not in data or 'password' not in data:
            return jsonify({"status": "error", "message": "Email and password are required"}), 400
        
        email = data['email']
        password = data['password']

        if not is_valid_email(email):
            return jsonify({"message": "Invalid email"}), 400
        
        # Check if user already exists
        existing_user = find_user_by_email(email)
        if existing_user:
            return jsonify({"status": "error", "message": "User with this email already exists"}), 409
        
        # Create new user
        connection = create_db_connection()
        if connection is None:
            return jsonify({"status": "error", "message": "Database connection failed"}), 500
        
        try:
            # Hash the password
            hashed_password = generate_password_hash(password)
            
            # Insert user into database (without token initially)
            query = "INSERT INTO users (email, password, created_at) VALUES (%s, %s, NOW())"
            cursor = execute_query(connection, query, (email, hashed_password), fetch=False)  # No need to fetch anything here

            # Get the new user ID after insert
            user_id = cursor.lastrowid  # Access lastrowid from the cursor object

            # Generate token
            token = create_token(user_id)
            
            # Update user record with token
            update_query = "UPDATE users SET token = %s WHERE id = %s"
            execute_query(connection, update_query, (token, user_id), fetch=False)
            
            return jsonify({
                "status": "success",
                "message": "User registered successfully",
                "token": token,
                "user": {
                    "id": user_id,
                    "email": email
                }
            }), 200
            
        except Error as e:
            return jsonify({"status": "error", "message": f"Database error: {str(e)}"}), 500
        finally:
            if connection.is_connected():
                connection.close()
                
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500



# Updated Login API
@app.route('/api/login', methods=['POST'])
def login():
    try:
        data = request.json
        
        # Validate request data
        if not data or 'email' not in data or 'password' not in data:
            return jsonify({"status": "error", "message": "Email and password are required"}), 400
        
        email = data['email']
        password = data['password']
        
        # Find user
        user = find_user_by_email(email)
        if not user:
            return jsonify({"status": "error", "message": "Invalid email or password"}), 400
        
        # Check password
        print(f"{user['password']}")
        if not check_password_hash(user['password'], password):
            return jsonify({"status": "error", "message": "Invalid email or password"}), 400
        
        connection = create_db_connection()
        if connection is None:
            return jsonify({"status": "error", "message": "Database connection failed"}), 500
            
        try:
            # Create JWT token
            token = create_token(user['id'])
            
            # Update token in database
            update_query = "UPDATE users SET token = %s WHERE id = %s"
            execute_query(connection, update_query, (token, user['id']))
            
            return jsonify({
                "status": "success",
                "message": "Login successful",
                "token": token,
                "user": {
                    "id": user['id'],
                    "email": user['email']
                }
            })
        except Error as e:
            return jsonify({"status": "error", "message": f"Database error: {str(e)}"}), 500
        finally:
            if connection.is_connected():
                connection.close()
        
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

# New function to verify JWT token
def verify_token(token):
    try:
        payload = pyjwt.decode(token, JWT_SECRET_KEY, algorithms=['HS256'])
        return payload['sub']  # Return user_id
    except pyjwt.ExpiredSignatureError:
        return None  # Token expired
    except pyjwt.InvalidTokenError:
        return None  # Invalid token

# New function to get user by ID
def get_user_by_id(user_id):
    connection = create_db_connection()
    if connection is None:
        return None
    
    try:
        query = "SELECT id, email, created_at FROM users WHERE id = %s"
        cursor = execute_query(connection, query, (user_id,))
        user = cursor.fetchone()
        return user
    except Error as e:
        print(f"Error finding user: {e}")
        return None
    finally:
        if connection.is_connected():
            connection.close()

# New User Details API
@app.route('/api/user/details', methods=['GET'])
def user_details():
    # Get token from Authorization header
    auth_header = request.headers.get('Authorization')
    if not auth_header or not auth_header.startswith('Bearer '):
        return jsonify({"status": "error", "message": "Authorization token required"}), 401
    
    token = auth_header.split(' ')[1]
    
    # Verify token
    user_id = verify_token(token)
    if not user_id:
        return jsonify({"status": "error", "message": "Invalid or expired token"}), 401
    
    # Get user details
    user = get_user_by_id(user_id)
    if not user:
        return jsonify({"status": "error", "message": "User not found"}), 404
    
    # Format the created_at date for better readability
    created_at = user['created_at']
    if isinstance(created_at, datetime.datetime):
        created_at = created_at.strftime('%Y-%m-%d %H:%M:%S')
    
    return jsonify({
        "status": "success",
        "user": {
            "id": user['id'],
            "email": user['email'],
            "created_at": created_at
        }
    })

if __name__ == '__main__':
    app.run(debug=True, host = "0.0.0.0",port=5000)
