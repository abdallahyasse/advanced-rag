from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch

_model = None
_tokenizer = None

def get_model():
    global _model, _tokenizer
    if _model is None:
        _tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-base")
        _model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-base")
        _model.eval()
    return _model, _tokenizer

def generate_answer(context: str, question: str) -> str:
    model, tokenizer = get_model()
    prompt = f"Answer the question based on the context.\n\nContext: {context[:1000]}\n\nQuestion: {question}\n\nAnswer:"
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=100)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)