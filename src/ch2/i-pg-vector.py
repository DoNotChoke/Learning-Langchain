from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_postgres.vectorstores import PGVector
from langchain_core.documents import Document
import uuid

connection = "postgresql+psycopg://langchain:langchhain@localhost:6024/langchain"

raw_documents = PyPDFLoader("Retrofitting Large Language Models with Dynamic Tokenization.pdf", extract_images=True).load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
documents = text_splitter.split_documents(raw_documents)

embeddings_model = HuggingFaceEmbeddings()
db = PGVector.from_documents(
    documents, embedding=embeddings_model, connection=connection
)
results = db.similarity_search("query", k=4)

print(results)

print("Adding documents to the vector store")
ids = [str(uuid.uuid4()), str(uuid.uuid4())]

db.add_documents(
    [
        Document(
            page_content="there are cats in the pond",
            metadata={"location": "pond", "topic": "animals"},
        ),
        Document(
            page_content="ducks are also found in the pond",
            metadata={"location": "pond", "topic": "animals"},
        ),
    ],
    ids=ids,
)

print("Documents added successfully.\n Fetched documents count:",
      len(db.get_by_ids(ids)))

print("Deleting document with id", ids[1])
db.delete(ids=[ids[1]])

print("Document deleted successfully.\n Fetched documents count:",
      len(db.get_by_ids(ids)))
