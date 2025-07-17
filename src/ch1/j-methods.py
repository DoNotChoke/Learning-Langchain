from src.model import get_model_together

model = get_model_together()

completion = model.invoke("Hi there!")

completions = model.batch(["Hi there!", "Bye!"])

for token in model.stream("Bye!"):
    print(token.content)