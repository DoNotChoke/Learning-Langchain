from pydantic import BaseModel, Field
from src.model import get_model_together

class Joke(BaseModel):
    setup: str = Field(description="The setup of the joke")
    punchline: str = Field(description="The punchline to the joke")

model = get_model_together()
model = model.with_structured_output(Joke)

result = model.invoke("Tell me a joke about cats")
print(result)