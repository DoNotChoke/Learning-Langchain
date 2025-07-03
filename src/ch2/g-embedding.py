from langchain_huggingface import HuggingFaceEmbeddings

embed_model = HuggingFaceEmbeddings()

embeddings = embed_model.embed_documents([
    "Hi there!",
    "Oh, hello!",
    "What's your name?",
    "My friends call me World",
    "Hello World!"
])

print(embeddings)