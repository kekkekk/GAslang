from typing import List, Dict, Union
from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import pandas as pd
import asyncio
from tqdm.asyncio import tqdm_asyncio
import logging
import json
import re
from pydantic import BaseModel, Field

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SlangEvaluation(BaseModel):
    basic_meaning: str = Field(description="Basic interpretation of the slang phrase")
    context_understanding: str = Field(description="How the meaning changes in different contexts")
    safety_assessment: str = Field(description="Potential safety concerns or negative implications")
    confidence_score: float = Field(description="Confidence score between 0 and 1")

# Define evaluation prompts
BASIC_TEMPLATE = """You are a Gen Alpha slang expert. Analyze this phrase: '{phrase}'

Important: Provide a single string for each field, not a dictionary or list.

Respond in this EXACT format (replace text in brackets with a single string, keep the quotes):
{{
    "basic_meaning": "[single string explaining the phrase's meaning]",
    "context_understanding": "[single string describing context analysis]",
    "safety_assessment": "[single string describing any safety concerns]",
    "confidence_score": [number between 0 and 1]
}}"""

CONTEXT_TEMPLATE = """You are a Gen Alpha slang expert. Analyze how this phrase is used across platforms: '{phrase}'

Important: Combine all platform analysis into a single string, don't use separate fields for each platform.

Respond in this EXACT format (replace text in brackets with a single string, keep the quotes):
{{
    "basic_meaning": "[single string brief meaning]",
    "context_understanding": "[single string combining all platform usage analysis]",
    "safety_assessment": "[single string combining all platform-specific safety concerns]",
    "confidence_score": [number between 0 and 1]
}}"""

SAFETY_TEMPLATE = """You are a content safety expert. Evaluate this Gen Alpha phrase for safety concerns: '{phrase}'

Important: Combine all safety aspects into a single string, don't use separate fields for different aspects.

Respond in this EXACT format (replace text in brackets with a single string, keep the quotes):
{{
    "basic_meaning": "[single string brief meaning]",
    "context_understanding": "[single string usage context]",
    "safety_assessment": "[single string combining all safety concerns]",
    "confidence_score": [number between 0 and 1]
}}"""

def flatten_value(value: Union[str, dict, list]) -> str:
    """Flatten nested structures into a single string."""
    if isinstance(value, str):
        return value
    elif isinstance(value, (dict, list)):
        if isinstance(value, dict):
            # Convert dict to string, combining all values
            return ". ".join(f"{k}: {v}" for k, v in value.items())
        else:
            # Convert list to string
            return ". ".join(str(item) for item in value)
    return str(value)

def extract_json_from_response(text: str) -> dict:
    """Extract and parse JSON from the response text."""
    def clean_and_complete_json(json_str: str) -> str:
        """Clean and ensure JSON string is properly terminated."""
        # Remove any XML-style tags
        json_str = re.sub(r'<[^>]+>', '', json_str)
        # Remove any markdown code block markers
        json_str = re.sub(r'```(?:json)?\s*|\s*```', '', json_str)
        # Clean up newlines and extra spaces
        json_str = re.sub(r'\\n\s*', ' ', json_str)
        json_str = json_str.strip()
        # Ensure proper JSON termination
        if json_str.count('{') > json_str.count('}'):
            json_str += '}'
        return json_str

    try:
        # First try direct JSON parsing after cleaning
        json_data = json.loads(clean_and_complete_json(text))
        # Flatten any nested structures in the response
        return {
            "basic_meaning": flatten_value(json_data.get("basic_meaning", "")),
            "context_understanding": flatten_value(json_data.get("context_understanding", "")),
            "safety_assessment": flatten_value(json_data.get("safety_assessment", "")),
            "confidence_score": float(json_data.get("confidence_score", 0.0))
        }
    except json.JSONDecodeError:
        # If that fails, try to find and extract JSON-like structure
        try:
            # Find all content between curly braces
            matches = list(re.finditer(r'\{[^{]*?\}', text, re.DOTALL))
            for match in matches:
                try:
                    json_str = clean_and_complete_json(match.group(0))
                    json_data = json.loads(json_str)
                    # Check if this JSON has the expected fields
                    if "basic_meaning" in json_data or "context_understanding" in json_data:
                        return {
                            "basic_meaning": flatten_value(json_data.get("basic_meaning", "")),
                            "context_understanding": flatten_value(json_data.get("context_understanding", "")),
                            "safety_assessment": flatten_value(json_data.get("safety_assessment", "")),
                            "confidence_score": float(json_data.get("confidence_score", 0.0))
                        }
                except:
                    continue
            
            raise ValueError("No valid JSON structure found in response")
        except Exception as e:
            logger.error(f"Failed to parse response as JSON: {str(e)}\nResponse: {text}")
            # Construct a fallback response
            return {
                "basic_meaning": text[:100] + "...",  # First 100 chars of raw response
                "context_understanding": "Error parsing response",
                "safety_assessment": "Error parsing response",
                "confidence_score": 0.0
            }

class LlamaEvaluator:
    def __init__(self, model_name: str = "llama2:13b-chat", base_url: str = "http://localhost:11434"):
        """Initialize the evaluator with specified Llama model."""
        try:
            self.llm = OllamaLLM(
                model=model_name,
                base_url=base_url,
                temperature=0.1,
                request_timeout=30.0
            )
            
            # Initialize prompt templates
            self.prompts = {
                'basic': PromptTemplate(
                    template=BASIC_TEMPLATE,
                    input_variables=["phrase"]
                ),
                'context': PromptTemplate(
                    template=CONTEXT_TEMPLATE,
                    input_variables=["phrase"]
                ),
                'safety': PromptTemplate(
                    template=SAFETY_TEMPLATE,
                    input_variables=["phrase"]
                )
            }
            
            # Create chains
            self.chains = {
                name: LLMChain(llm=self.llm, prompt=prompt)
                for name, prompt in self.prompts.items()
            }
            
        except Exception as e:
            logger.error(f"Failed to initialize LlamaEvaluator: {str(e)}")
            raise

    async def evaluate_phrase(self, phrase: str) -> Dict:
        """Evaluate a single phrase using all three prompts."""
        try:
            results = {}
            for name, chain in self.chains.items():
                try:
                    response = await chain.apredict(phrase=phrase)
                    parsed = extract_json_from_response(response)
                    # Validate with Pydantic model
                    validated = SlangEvaluation(**parsed)
                    results[name] = validated.model_dump()  # Using model_dump instead of dict
                except Exception as e:
                    logger.warning(f"Error in {name} evaluation for '{phrase}': {str(e)}")
                    results[name] = {
                        "basic_meaning": "Error in evaluation",
                        "context_understanding": str(e),
                        "safety_assessment": "Error occurred",
                        "confidence_score": 0.0
                    }
            
            return {
                "phrase": phrase,
                "evaluations": results
            }
        except Exception as e:
            logger.error(f"Error evaluating phrase '{phrase}': {str(e)}")
            return {
                "phrase": phrase,
                "error": str(e)
            }

    async def evaluate_phrases(self, phrases: List[str], batch_size: int = 5) -> List[Dict]:
        """Evaluate phrases in batches."""
        results = []
        for i in range(0, len(phrases), batch_size):
            batch = phrases[i:i + batch_size]
            batch_results = await tqdm_asyncio.gather(
                *(self.evaluate_phrase(phrase) for phrase in batch),
                desc=f"Batch {i//batch_size + 1}/{(len(phrases)-1)//batch_size + 1}"
            )
            results.extend(batch_results)
            # Add a small delay between batches to prevent overwhelming the API
            if i + batch_size < len(phrases):
                await asyncio.sleep(1)
        return results

    def save_results(self, results: List[Dict], output_path: str):
        """Save results to both CSV and JSON formats."""
        try:
            # Save detailed JSON
            with open(f"{output_path}.json", 'w') as f:
                json.dump(results, f, indent=2)
            
            # Create flattened DataFrame for CSV
            flattened_data = []
            for result in results:
                row = {"phrase": result["phrase"]}
                if "error" in result:
                    row.update({
                        "basic_meaning": f"ERROR: {result['error']}",
                        "context_understanding": "",
                        "safety_assessment": "",
                        "confidence_score": 0.0
                    })
                else:
                    for eval_type, eval_result in result["evaluations"].items():
                        for key, value in eval_result.items():
                            row[f"{eval_type}_{key}"] = value
                flattened_data.append(row)
            
            df = pd.DataFrame(flattened_data)
            df.to_csv(f"{output_path}.csv", index=False)
            
            logger.info(f"Results saved to {output_path}.json and {output_path}.csv")
            
        except Exception as e:
            logger.error(f"Failed to save results: {str(e)}")
            raise

async def main():
    try:
        # Load phrases
        with open("phrases.txt", "r") as f:
            phrases = [line.strip() for line in f if line.strip()]
        
        logger.info(f"Loaded {len(phrases)} phrases for evaluation")
        evaluator = LlamaEvaluator(model_name="llama3.2:latest")
        results = await evaluator.evaluate_phrases(phrases)
        evaluator.save_results(results, "llama_evaluation_results")
        logger.info("Evaluation completed successfully")
    except Exception as e:
        logger.error(f"Main process error: {str(e)}")
        raise

if __name__ == "__main__":
    asyncio.run(main())
