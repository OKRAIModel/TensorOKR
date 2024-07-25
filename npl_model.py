from transformers import GPT2Tokenizer, GPT2LMHeadModel
from transformers import DebertaTokenizer, DebertaModel
from transformers import RobertaTokenizer, RobertaModel

def load_nlp_models():
    # Tải mô hình GPT-2
    gpt2_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    gpt2_model = GPT2LMHeadModel.from_pretrained('gpt2')
    
    # Tải mô hình DeBERTa
    deberta_tokenizer = DebertaTokenizer.from_pretrained('microsoft/deberta-base')
    deberta_model = DebertaModel.from_pretrained('microsoft/deberta-base')
    
    # Tải mô hình RoBERTa
    roberta_tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    roberta_model = RobertaModel.from_pretrained('roberta-base')
    
    return (gpt2_tokenizer, gpt2_model), (deberta_tokenizer, deberta_model), (roberta_tokenizer, roberta_model)

def generate_text_gpt2(tokenizer, model, prompt):
    inputs = tokenizer.encode(prompt, return_tensors='pt')
    outputs = model.generate(inputs, max_length=50, num_return_sequences=1)
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return text

def encode_text(tokenizer, model, text):
    inputs = tokenizer(text, return_tensors='pt')
    outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1)