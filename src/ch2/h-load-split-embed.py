from langchain_community.document_loaders import TextLoader

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings

loader = TextLoader("../test.txt", encoding="utf-8")
doc = loader.load()

splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.split_documents(doc)

embed_model = HuggingFaceEmbeddings()
embeddings = embed_model.embed_documents(
    [chunk.page_content for chunk in chunks]
)

print(embeddings)