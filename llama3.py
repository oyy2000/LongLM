import SelfExtend
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from datasets import load_dataset
from tqdm import tqdm

# Load your model, e.g., loaded_model = AutoModelForCausalLM.from_pretrained(model_path)
# model_lists = ['google/gemma-7b-it', 'meta-llama/Llama-2-7b-chat-hf', 'mistralai/Mistral-7B-Instruct-v0.1', ]
model_path = 'meta-llama/Meta-Llama-3-8B'
model = AutoModelForCausalLM.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# group size, neighbor window.
group_size = 1024
window_size = 32

SelfExtend.apply(model, group_size, window_size, enable_flash_attention=False)

# Load the IMDb movie reviews dataset
dataset = load_dataset("imdb")

# Filter reviews with more than 1400 characters
long_reviews = dataset['test'].filter(lambda x: len(x['text']) > 4000)

def predict_sentiment(text):
    # Encode the text
    inputs = tokenizer(text, return_tensors="pt")
    # Generate output using the model
    outputs = model.generate(**inputs)
    # Decode the output
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Determine sentiment
    if "positive" in result.lower():
        return "positive"
    elif "negative" in result.lower():
        return "negative"
    else:
        return "neutral"

# Compute success rate
correct_predictions = 0
total_reviews = len(long_reviews)


# Compute success rate
correct_predictions = 0
total_reviews = len(long_reviews)

for review, label in tqdm(zip(long_reviews['text'], long_reviews['label']), total=total_reviews):
    predicted_sentiment = predict_sentiment(review)
    # Convert label to string ('positive' or 'negative')
    actual_sentiment = 'positive' if label == 1 else 'negative'
    if predicted_sentiment == actual_sentiment:
        correct_predictions += 1

success_rate = correct_predictions / total_reviews * 100
print("Success rate: {:.2f}%".format(success_rate))