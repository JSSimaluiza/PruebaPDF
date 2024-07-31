# Importa la función load_dotenv del módulo dotenv para cargar variables de entorno desde un archivo .env
from dotenv import load_dotenv
# Importa el módulo os para interactuar con el sistema operativo  
import os
# Importa la clase PdfReader del módulo PyPDF2 para leer archivos PDF  
from PyPDF2 import PdfReader
# Importa el módulo Flask para crear una aplicación web
from flask import Flask, request, jsonify
# Importa el CharacterTextSplitter del módulo langchain.text_splitter para dividir texto en caracteres
from langchain.text_splitter import CharacterTextSplitter  
# Importa OpenAIEmbeddings del módulo langchain_community.embeddings.openai para generar incrustaciones de texto utilizando OpenAI
from langchain_community.embeddings import OpenAIEmbeddings  
# Importa FAISS del módulo langchain_community.vectorstores para realizar búsqueda de similitud
from langchain_community.vectorstores import FAISS  
# Importa load_qa_chain del módulo langchain.chains.question_answering para cargar cadenas de preguntas y respuestas
from langchain.chains.question_answering import load_qa_chain  
# Importa OpenAI del módulo langchain_community.llms para interactuar con el modelo de lenguaje de OpenAI
from langchain_community.llms import OpenAI  
# Importa get_openai_callback del módulo langchain_community.callbacks.manager para obtener realimentación de OpenAI
from langchain_community.callbacks.manager import get_openai_callback  
# Importa el módulo langchain
import langchain

# Desactiva la salida detallada de la biblioteca langchain
langchain.verbose = False  

# Carga las variables de entorno desde un archivo .env
load_dotenv()

app = Flask(__name__)

# Función para procesar el texto extraído de un archivo PDF
def process_text(text):
    # Divide el texto en trozos usando langchain
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )

    chunks = text_splitter.split_text(text)

    # Convierte los trozos de texto en incrustaciones para formar una base de conocimientos
    embeddings = OpenAIEmbeddings(openai_api_key=os.environ.get("OPENAI_API_KEY"))

    knowledge_base = FAISS.from_texts(chunks, embeddings)

    return knowledge_base

@app.route('/upload', methods=['POST'])
def upload_pdf():
    # Verifica si se ha subido un archivo
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    file = request.files['file']
    
    # Verifica si el archivo tiene un nombre
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    pdf_reader = PdfReader(file)
    
    # Almacena el texto del PDF en una variable
    text = ""

    for page in pdf_reader.pages:
        text += page.extract_text()

    # Crea un objeto de base de conocimientos a partir del texto del PDF
    knowledge_base = process_text(text)

    return jsonify({"knowledge_base": knowledge_base}), 200

@app.route('/ask', methods=['POST'])
def ask_question():
    data = request.get_json()
    query = data.get('question')
    knowledge_base_data = data.get('knowledge_base')
    
    # Verifica que se haya proporcionado una pregunta y una base de conocimientos
    if not query or not knowledge_base_data:
        return jsonify({"error": "Invalid input"}), 400

    knowledge_base = FAISS.from_existing_index(knowledge_base_data, OpenAIEmbeddings(openai_api_key=os.environ.get("OPENAI_API_KEY")))
    
    docs = knowledge_base.similarity_search(query)
    
    # Inicializa un modelo de lenguaje de OpenAI y ajustamos sus parámetros
    model = "gpt-3.5-turbo-instruct"  # Acepta 4096 tokens
    temperature = 0  # Valores entre 0 - 1
    
    llm = OpenAI(openai_api_key=os.environ.get("OPENAI_API_KEY"), model_name=model, temperature=temperature)
    chain = load_qa_chain(llm, chain_type="stuff")
    
    # Obtiene la realimentación de OpenAI para el procesamiento de la cadena
    with get_openai_callback() as cost:
        response = chain.invoke(input={"question": query, "input_documents": docs})
        print(cost)  # Imprime el costo de la operación
        
        return jsonify({"response": response["output_text"]}), 200

# Punto de entrada para la ejecución del programa
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)