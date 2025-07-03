from langchain_core.output_parsers import CommaSeparatedListOutputParser

parser = CommaSeparatedListOutputParser()

response = parser.invoke("Dog, Cat, Lion")

print(response)