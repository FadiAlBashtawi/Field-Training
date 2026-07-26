from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "distilgpt2"

print("Loading model...")

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

print("Model ready!")

while True:
    prompt = input("\nYou: ")

    if prompt.lower() == "exit":
        break

    inputs = tokenizer(prompt, return_tensors="pt")

    outputs = model.generate(
        **inputs,
        max_length=100,
        do_sample=True,
        temperature=0.7,
        top_p=0.9
    )

    response = tokenizer.decode(
        outputs[0],
        skip_special_tokens=True
    )

    print("\nAI:", response)