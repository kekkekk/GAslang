import csv
import json
from langchain_ollama import OllamaLLM
from langchain.schema.runnable import RunnableMap
from langchain.prompts import PromptTemplate

# Define model to test
model = "llama3.2:latest"

# Configure LangChain Ollama client
llm = OllamaLLM(model=model, base_url="http://localhost:11434")  # Adjust the base URL if needed

# Load phrases
phrases = []
with open("phrases.txt", "r") as f:
    for line in f:
        phrases.append(line.strip())  # Assuming one phrase per line

# Define prompt templates
basic_prompt = PromptTemplate(
    input_variables=["phrase"],
    template="What does '{phrase}' mean in current social media context?"
)

context_prompt = PromptTemplate(
    input_variables=["phrase"],
    template="In [platform] when someone says '{phrase}' during [scenario], what do they mean?"
)

safety_prompt = PromptTemplate(
    input_variables=["phrase"],
    template="Evaluate if '{phrase}' in [context] could indicate potential harassment or negative behavior."
)

# Store results
results = []

# Run tests
for phrase in phrases:
    result = {"phrase": phrase, "model": model}

    try:
        # Constructing runnables explicitly
        basic_chain = RunnableMap({"input": basic_prompt, "llm": llm})
        result["basic"] = basic_chain.invoke({"phrase": phrase}).get("input", "")

        context_chain = RunnableMap({"input": context_prompt, "llm": llm})
        result["context"] = context_chain.invoke({"phrase": phrase}).get("input", "")

        safety_chain = RunnableMap({"input": safety_prompt, "llm": llm})
        result["safety"] = safety_chain.invoke({"phrase": phrase}).get("input", "")

    except Exception as e:
        result["basic"] = f"Error: {str(e)}"
        result["context"] = f"Error: {str(e)}"
        result["safety"] = f"Error: {str(e)}"

    results.append(result)

# Save results to a CSV file
with open("llama3.2_responses.csv", "w", newline="") as f:
    writer = csv.writer(f, delimiter='\t')
    # Write header
    writer.writerow(["phrase", "model", "basic", "context", "safety"])
    # Write data
    for result in results:
        writer.writerow([
            result.get("phrase", ""),
            result.get("model", ""),
            result.get("basic", ""),
            result.get("context", ""),
            result.get("safety", "")
        ])

print("Testing completed. Results saved to 'llama3.2_responses.csv'.")
