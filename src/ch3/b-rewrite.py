from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_postgres.vectorstores import PGVector
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import chain
from langchain_huggingface import HuggingFaceEmbeddings
from src.model import get_model_together


connection = "postgresql+psycopg://langchain:langchhain@localhost:6024/langchain"

raw_documents = TextLoader("../test.txt", encoding = "utf-8").load()

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

query = 'Today I woke up and brushed my teeth, then I sat down to read the news. But then I forgot the food on the cooker. Who are some key figures in the ancient greek history of philosophy?'

docs = retriever.invoke(query)


print(docs[0].page_content)
print("\n\n")

prompt = ChatPromptTemplate.from_template(
    """Answer the question based only on the following context: {context} Question: {question} """
)

llm = get_model_together()

@chain
def qa(input):
    docs = retriever.invoke(input)

    formatted = prompt.invoke({"context": docs, "question": input})
    answer = llm.invoke(formatted)

    return answer

result = qa.invoke(query)

print(result.content)

print("\nRewrite the query to improve accuracy\n")

rewrite_prompt = ChatPromptTemplate.from_template(
    """Provide a better search query for web search engine to answer the given question, end the queries with ’**’. Question: {x} Answer:""")


def parse_rewriter_output(message):
    return message.content.strip('"').strip("**")

rewriter = rewrite_prompt | llm | parse_rewriter_output

@chain
def qa_rrr(input):
    new_query = rewriter.invoke(input)
    print("Rewritten query: ", new_query)
    # fetch relevant documents
    docs = retriever.invoke(new_query)
    # format prompt
    formatted = prompt.invoke({"context": docs, "question": input})
    # generate answer
    answer = llm.invoke(formatted)
    return answer

print("\nCall model again with rewritten query\n")

# call model again with rewritten query
result = qa_rrr.invoke(query)
print(result.content)
