from src.model import get_model

model = get_model()

completion = model.invoke("Hi there!")

completions = model.batch(["Hi there!", "Bye!"])

for token in model.stream("Bye!"):
    print(token.content)