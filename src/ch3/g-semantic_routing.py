from langchain.utils.math import cosine_similarity
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import chain
from langchain_huggingface import HuggingFaceEmbeddings
from src.model import get_model_together

physics_template = """You are a physics professor. 
                      Your task is to answer question about physics in a concise and easy-to-understand manner.
                       When you don't know the answer to a question, you admit that you don't know. 
                       Here is a question: {query}
                    """

math_template = """You are a mathematician.
                    You are great at answering math questions. 
                    You are so good because you are able to break down hard problems into their component parts, answer the component parts, and then put them together to answer the broader question.
                    Here is a question: {query}
                """

embeddings_model = HuggingFaceEmbeddings()
prompt_templates = [physics_template, math_template]

prompt_embeddings = embeddings_model.embed_documents(prompt_templates)

@chain
def prompt_router(query):
    query_embedding = embeddings_model.embed_query(query)
    similarity = cosine_similarity([query_embedding], prompt_embeddings)[0]
    most_similar = prompt_templates[similarity.argmax()]
    print("Using MATH" if most_similar == math_template else "Using PHYSICS")
    return PromptTemplate.from_template(most_similar)

llm = get_model_together()

semantic_router = (prompt_router | llm | StrOutputParser())

result = semantic_router.invoke("What's a black hole")
print("\nSemantic router result: ", result)