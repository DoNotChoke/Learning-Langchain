from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_postgres.vectorstores import PGVector
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.runnables import chain
from src.model import get_model_together

connection = "postgresql+psycopg://langchain:langchhain@localhost:6024/langchain"

raw_documents = TextLoader("../test.txt", encoding="utf-8").load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

documents = text_splitter.split_documents(raw_documents)

embeddings_model = HuggingFaceEmbeddings()

db = PGVector.from_documents(
    documents,
    embedding=embeddings_model,
    connection=connection,
)

retriever = db.as_retriever(search_kwargs={"k": 2})

query = 'Who are the key figures in the ancient greek history of philosophy?'

docs = retriever.invoke(query)

print(docs[0].page_content)

prompt = ChatPromptTemplate.from_template(
    """Answer the question based only on the following context: {context} Question: {question} """
)

llm = get_model_together()

llm_chain = prompt | llm

result = llm_chain.invoke({"context": docs, "question": query})

print(result)
print("\n\n")

print("Running again but this time encapsulate the logic for efficiency\n")

@chain
def qa(input):
    context = retriever.invoke(input)

    formatted = prompt.invoke({"context": context, "question": input})

    answer = llm.invoke(formatted)
    return answer

result = qa.invoke(query)
print(result.content)