
from uuid import uuid4
from pathlib import Path
from dotenv import load_dotenv
from langchain.chains import RetrievalQAWithSourcesChain
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain_chroma import Chroma
from langchain_huggingface.embeddings import HuggingFaceEmbeddings

load_dotenv()

# Constants
CHUNK_SIZE = 1000
EMBEDDING_MODEL = "Alibaba-NLP/gte-base-en-v1.5"
COLLECTION_NAME = 'real_estate'
VECTOR_STROE_DIR = Path(__file__).parent / "resources/vectorstore"

llm = None
vector_store = None

#Initialize LLM, Vector db
def initialize_components():
    
    global llm, vector_store

    if llm is None:
        llm = ChatGroq(model = 'llama-3.3-70b-versatile', temperature=0.9, max_tokens=500)
    
    if vector_store is None:
        ef = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"trust_remote_code":True,"device": "cpu"}
        )

        
        vector_store = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=ef,
            persist_directory=str(VECTOR_STROE_DIR)
        )

#Scrap data from the url into a vector db
def process_urls(urls):
    
    yield "Initializing Components.."
    initialize_components()

    yield "Resetting the vector db.."
    vector_store.reset_collection()

    yield "Loading Data.."
    #Document Loader
    loader = UnstructuredURLLoader(urls = urls)
    data = loader.load()

    yield "Text Splitting.."
    #Text Splitter
    text_splitter = RecursiveCharacterTextSplitter(
        separators=['\n\n','\n','.',' '],
        chunk_size = CHUNK_SIZE,
    )
    docs = text_splitter.split_documents(data)

    yield "Adding docs to vector db.."
    #Generate unique Ids
    uuids = [str(uuid4()) for _ in range(len(docs))]
    vector_store.add_documents(docs,ids=uuids)

    yield "Done adding docs to the vector db ✅"

#Generate Answer from query asked
def generate_answer(query):
    if not vector_store:
        raise RuntimeError("Vector db is not initialized")
    
    chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever = vector_store.as_retriever())
    result = chain.invoke({"question":query},return_only_outputs=True)
    sources = result.get("sources","")

    return result['answer'], sources

