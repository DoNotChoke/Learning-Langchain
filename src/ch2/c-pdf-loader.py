from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("Retrofitting Large Language Models with Dynamic Tokenization.pdf")
pages = loader.load()

print(pages)