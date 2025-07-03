from langchain.indexes import SQLRecordManager, index
from langchain_postgres.vectorstores import PGVector
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.docstore.document import Document

connection = "postgresql+psycopg://langchain:langchhain@localhost:6024/langchain"
collection_name = "my_docs"
embeddings_model = HuggingFaceEmbeddings()
namespace = "my_docs_namespace"

vectorstore = PGVector(
    embeddings=embeddings_model,
    collection_name=collection_name,
    connection=connection,
    use_jsonb=True
)

record_manager = SQLRecordManager(
    namespace,
    db_url=connection
)

record_manager.create_schema()

docs = [
    Document(page_content='there are cats in the pond', metadata={
             "id": 1, "source": "cats.txt"}),
    Document(page_content='ducks are also found in the pond', metadata={
             "id": 2, "source": "ducks.txt"}),
]

index_1 = index(
    docs,
    record_manager,
    vectorstore,
    cleanup="incremental", # prevent duplicate documents
    source_id_key="source" # use the source field as the source_id
)

print("Index attempt 1:", index_1)

# second time you attempt to index, it will not add the documents again
index_2 = index(
    docs,
    record_manager,
    vectorstore,
    cleanup="incremental",
    source_id_key="source",
)

print("Index attempt 2:", index_2)

# If we mutate a document, the new version will be written and all old versions sharing the same source will be deleted.

docs[0].page_content = "I just modified this document!"

index_3 = index(
    docs,
    record_manager,
    vectorstore,
    cleanup="incremental",
    source_id_key="source",
)

print("Index attempt 3:", index_3)