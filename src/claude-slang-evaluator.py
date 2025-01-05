import os
import json
import time
import pandas as pd
from abc import ABC, abstractmethod
from typing import List, Dict, Optional
from datetime import datetime
from anthropic import Anthropic
from openai import OpenAI
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from huggingface_hub import HfApi
import google.generativeai as genai

def clean_value(value):
    """Clean and format values for TSV output"""
    if value is None:
        return ""
    if isinstance(value, (int, float)):
        return value
    return str(value).strip().replace('\t', ' ').replace('\n', ' ')

class LLMEvaluator(ABC):
    """Abstract base class for LLM evaluators"""
    
    def __init__(self):
        self.base_prompt = """Analyze this term: "{phrase}"

Return a JSON object exactly like this (replace the values but keep the exact format):
{
"interpretation": "write what the term means and how it's used",
"concern_level": 0,
"moderation_action": "what action you recommend as a content moderator",
"confidence": 0.5
}

Important: Return only the JSON with no additional text or formatting."""
    
    @abstractmethod
    def evaluate_phrase(self, phrase: str) -> Dict:
        """Evaluate a single phrase using the LLM"""
        pass
    
    def evaluate_phrases(self, phrases: List[str], output_file: Optional[str] = None) -> pd.DataFrame:
        """Evaluate multiple phrases and save results"""
        results = []
        
        for phrase in phrases:
            try:
                result = self.evaluate_phrase(phrase)
                results.append(result)
                time.sleep(1)  # Rate limiting
                
            except Exception as e:
                print(f"Error evaluating phrase '{phrase}': {str(e)}")
                results.append({
                    'phrase': phrase,
                    'model': self.__class__.__name__,
                    'interpretation_type': 'error',
                    'interpretation_text': f"Error: {str(e)}",
                    'interpretation_flags': '',
                    'concern_level': None,
                    'moderation_action': '',
                    'confidence': 0,
                    'error': str(e)
                })
        
        # Convert to DataFrame with structured columns
        df = pd.DataFrame(results)
        
        # Save results if output file specified
        if output_file:
            df.to_csv(output_file, sep='\t', index=False, escapechar='\\')
        
        return df

class ClaudeEvaluator(LLMEvaluator):
    """Anthropic Claude API evaluator"""
    
    def __init__(self, api_key: str, model: str = "claude-3-sonnet-20240229"):
        super().__init__()
        self.client = Anthropic(api_key=api_key)
        self.model = model
    
    def evaluate_phrase(self, phrase: str) -> Dict:
        message = self.client.messages.create(
            model=self.model,
            max_tokens=1000,
            temperature=0,
            messages=[{
                "role": "user",
                "content": self.base_prompt.format(phrase=phrase)
            }]
        )
        
        try:
            parsed = json.loads(message.content[0].text)
            return {
                'phrase': phrase,
                'model': 'claude',
                'interpretation_type': 'standard',
                'interpretation_text': clean_value(parsed.get('interpretation', '')),
                'interpretation_flags': '',
                'concern_level': parsed.get('concern_level'),
                'moderation_action': clean_value(parsed.get('moderation_action', '')),
                'confidence': parsed.get('confidence', 0),
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            raise Exception(f"Failed to parse Claude response: {str(e)}")

class OpenAIEvaluator(LLMEvaluator):
    """OpenAI GPT API evaluator"""
    
    def __init__(self, api_key: str, model: str = "gpt-4"):
        super().__init__()
        self.client = OpenAI(api_key=api_key)
        self.model = model
    
    def evaluate_phrase(self, phrase: str) -> Dict:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{
                "role": "user",
                "content": self.base_prompt.format(phrase=phrase)
            }],
            temperature=0
        )
        
        try:
            parsed = json.loads(response.choices[0].message.content)
            return {
                'phrase': phrase,
                'model': 'openai',
                'interpretation_type': 'standard',
                'interpretation_text': clean_value(parsed.get('interpretation', '')),
                'interpretation_flags': '',
                'concern_level': parsed.get('concern_level'),
                'moderation_action': clean_value(parsed.get('moderation_action', '')),
                'confidence': parsed.get('confidence', 0),
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            raise Exception(f"Failed to parse OpenAI response: {str(e)}")

class GeminiEvaluator(LLMEvaluator):
    """Google Gemini API evaluator"""
    
    def __init__(self, api_key: str, model: str = "gemini-pro"):
        super().__init__()
        genai.configure(api_key=api_key)
        
        # Configure safety settings
        safety_settings = {
            "harassment": "block_none",
            "hate_speech": "block_none",
            "sexually_explicit": "block_none"
        }
        
        self.model = genai.GenerativeModel(
            model_name=model,
            safety_settings=safety_settings
        )
    
    def clean_json_text(self, text: str) -> str:
        """Clean text before JSON parsing"""
        # Print debug info
        print("Original text:", repr(text))
        
        # Remove leading/trailing whitespace and newlines
        text = text.strip()
        
        # Remove any leading/trailing quotes
        text = text.strip('"\'')
        
        # If text doesn't look like JSON, try to fix it
        if not text.startswith('{'):
            # Try to find JSON start
            start_idx = text.find('{')
            if start_idx != -1:
                text = text[start_idx:]
        
        if not text.endswith('}'):
            # Try to find JSON end
            end_idx = text.rfind('}')
            if end_idx != -1:
                text = text[:end_idx+1]
        
        # Replace escaped newlines and fix formatting
        text = text.replace('\\n', ' ')
        text = text.replace('\n', ' ')
        
        print("Cleaned text:", repr(text))
        return text
    
    def evaluate_phrase(self, phrase: str) -> Dict:
        try:
            response = self.model.generate_content(
                self.base_prompt.format(phrase=phrase),
                generation_config=genai.types.GenerationConfig(
                    temperature=0
                )
            )
            
            # Handle blocked responses
            if not response.candidates:
                return {
                    'phrase': phrase,
                    'model': 'gemini',
                    'interpretation_type': 'safety_flag',
                    'interpretation_text': '',
                    'interpretation_flags': 'Content blocked by safety filters',
                    'concern_level': 5,
                    'moderation_action': 'Content blocked',
                    'confidence': 1.0,
                    'timestamp': datetime.now().isoformat()
                }
            
            # Clean and parse response
            try:
                # Get response text and clean it
                text = response.text
                cleaned_text = self.clean_json_text(text)
                
                # First try direct parsing
                try:
                    parsed = json.loads(cleaned_text)
                except json.JSONDecodeError as e1:
                    print(f"First parse attempt failed: {str(e1)}")
                    # If that fails, try wrapping in curly braces
                    try:
                        if not cleaned_text.startswith('{'):
                            cleaned_text = '{' + cleaned_text
                        if not cleaned_text.endswith('}'):
                            cleaned_text = cleaned_text + '}'
                        parsed = json.loads(cleaned_text)
                    except json.JSONDecodeError as e2:
                        print(f"Second parse attempt failed: {str(e2)}")
                        raise e2
                
            except Exception as e:
                print(f"JSON parsing error for phrase '{phrase}': {str(e)}")
                return {
                    'phrase': phrase,
                    'model': 'gemini',
                    'interpretation_type': 'error',
                    'interpretation_text': f"Failed to parse response: {text[:100]}...",
                    'interpretation_flags': '',
                    'concern_level': None,
                    'moderation_action': '',
                    'confidence': 0,
                    'timestamp': datetime.now().isoformat()
                }
            
            # Return structured response
            return {
                'phrase': phrase,
                'model': 'gemini',
                'interpretation_type': 'standard',
                'interpretation_text': clean_value(parsed.get('interpretation', '')),
                'interpretation_flags': '',
                'concern_level': parsed.get('concern_level'),
                'moderation_action': clean_value(parsed.get('moderation_action', '')),
                'confidence': parsed.get('confidence', 0),
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            print(f"Error processing phrase '{phrase}': {str(e)}")
            return {
                'phrase': phrase,
                'model': 'gemini',
                'interpretation_type': 'error',
                'interpretation_text': f"Error: {str(e)}",
                'interpretation_flags': '',
                'concern_level': None,
                'moderation_action': '',
                'confidence': 0,
                'timestamp': datetime.now().isoformat()
            }

def read_phrases_from_file(filepath: str) -> List[str]:
    """Read phrases from text file, filtering out XML-style tags"""
    def is_valid_phrase(line: str) -> bool:
        line = line.strip()
        if not line:
            return False
        if line.startswith('<') and line.endswith('>'):
            return False
        if any(tag in line for tag in ['userStyle', 'userPreferences']):
            return False
        return True
        
    with open(filepath, 'r') as f:
        phrases = [line.strip() for line in f if is_valid_phrase(line.strip())]
    
    print(f"Found {len(phrases)} valid phrases")
    return phrases

def main():
    # Load environment variables for API keys
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")
    openai_key = os.getenv("OPENAI_API_KEY")
    gemini_key = os.getenv("GOOGLE_API_KEY")
    
    # Initialize evaluators
    evaluators = []
    
    #if anthropic_key:
    #    evaluators.append(ClaudeEvaluator(anthropic_key))
    
    if openai_key:
        evaluators.append(OpenAIEvaluator(openai_key))
        
    if gemini_key:
        evaluators.append(GeminiEvaluator(gemini_key))
    
    # Read phrases
    phrases = read_phrases_from_file("Gen1-10.txt")
    
    # Create output directory
    os.makedirs("llm_evaluations", exist_ok=True)
    
    # Evaluate with each model
    for evaluator in evaluators:
        model_name = evaluator.__class__.__name__
        output_file = os.path.join("llm_evaluations", f"{model_name}_results.tsv")
        
        print(f"\nEvaluating with {model_name}...")
        df = evaluator.evaluate_phrases(phrases, output_file)
        
        # Print statistics
        print(f"Processed {len(df)} phrases")
        print("\nColumn summary:")
        for col in df.columns:
            non_empty = df[col].notna().sum()
            print(f"- {col}: {non_empty} non-empty values")

if __name__ == "__main__":
    main()