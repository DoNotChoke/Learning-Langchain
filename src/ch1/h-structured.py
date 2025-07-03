from pydantic import BaseModel
from src.model import get_model

class AnswerWithJustification(BaseModel):
    """An answer to the user's question along with justification for the answer."""

    answer: str
    """The answer to the user's question"""
    justification: str
    """Justification for the answer"""

model = get_model()
structured_model = model.with_structured_output(AnswerWithJustification)

response = structured_model.invoke(
    "What weighs more, a pound of bricks or a pound of feathers")
print(response)
