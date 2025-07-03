from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("Retrofitting Large Language Models with Dynamic Tokenization.pdf")
docs = loader.load()

splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
splitted_docs = splitter.split_documents(docs)

print(splitted_docs)