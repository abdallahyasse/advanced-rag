from app.generation.generator_service import GeneratorService

context = """
Machine Learning is a branch of Artificial Intelligence.
"""

question = "What is Machine Learning?"

answer = GeneratorService.generate(
    context=context,
    question=question,
)

print(answer)