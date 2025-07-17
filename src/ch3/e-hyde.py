from langchain_community.document_loaders import TextLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_postgres.vectorstores import PGVector
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import chain
from langchain_core.output_parsers import StrOutputParser
from src.model import get_model_together

connection = "postgresql+psycopg://langchain:langchhain@localhost:6024/langchain"

raw_documents = TextLoader('../test.txt', encoding='utf-8').load()
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

retriever = db.as_retriever(search_kwargs={"k": 5})

prompt_hype = ChatPromptTemplate.from_template(
    """Please write a passage to answer the question.\n Question: {question} \n Passage:"""
)

llm = get_model_together()

generate_doc = (prompt_hype | llm | StrOutputParser())

retrieval_chain = generate_doc | retriever

query = "Who are some lesser known philosophers in the ancient greek history of philosophy?"

prompt = ChatPromptTemplate.from_template(
    """Answer the question based only on the following context: {context} Question: {question} """
)

@chain
def qa(input):
    docs = retrieval_chain.invoke(input)
    formatted = prompt.invoke({"context": docs, "question": input})
    answer = llm.invoke(formatted)
    return answer

print("Running hyde\n")
result = qa.invoke(query)
print("\n\n")
print(result.content)
