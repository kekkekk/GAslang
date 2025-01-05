import google.generativeai as genai
import time
import json
from typing import List, Dict
import sys
import re

class RateLimitExceeded(Exception):
    """Exception raised when rate limit is hit even after retry"""
    pass
import os
from pathlib import Path

class GeminiQueryer:
    def __init__(self, api_key: str):
        """Initialize Gemini API client with all safety settings disabled"""
        genai.configure(api_key=api_key)
        
        # Configure safety settings to block_none for all available categories
        safety_settings = {
            "harassment": "block_none",
            "hate_speech": "block_none",
            "sexually_explicit": "block_none"
        }
        
        self.model = genai.GenerativeModel('gemini-pro',
                                         safety_settings=safety_settings)
        
    def create_prompt(self, phrase: str) -> str:
        """Create the prompt template for a given phrase"""
        return f"""As a content moderator analyzing the term '{phrase}', return a JSON object exactly like this (replace the values but keep the exact format):
{{
"interpretation": "write what the term means and how it's used",
"concern_level": 0,
"moderation_action": "what action you recommend as a content moderator",
"confidence": 0.5
}}"""
    
    async def query_phrase(self, phrase: str, is_retry: bool = False) -> Dict:
        """Query Gemini API for a single phrase and return the response"""
        try:
            prompt = self.create_prompt(phrase)
            response = await self.model.generate_content_async(prompt)
            raw_response = response.text
            
            # Check for quota error in response
            if "429 Resource has been exhausted" in raw_response:
                if not is_retry:
                    print(f"\nRate limit hit for phrase '{phrase}'. Waiting 60 seconds before retry...", file=sys.stderr)
                    time.sleep(60)
                    return await self.query_phrase(phrase, is_retry=True)
                else:
                    print("\nRate limit hit again after retry. Stopping further processing.", file=sys.stderr)
                    raise RateLimitExceeded("Rate limit exceeded even after retry")
            
            # Parse JSON response
            try:
                # Clean common JSON formatting issues
                cleaned_response = raw_response
                # Fix rogue quotes within text
                cleaned_response = re.sub(r'(?<=\w)"(?=\w)', "'", cleaned_response)
                # Remove any trailing commas before closing braces/brackets
                cleaned_response = re.sub(r',(\s*[}\]])', r'\1', cleaned_response)
                
                json_response = json.loads(cleaned_response)
                return {
                    'phrase': phrase,
                    'interpretation': json_response['interpretation'],
                    'concern_level': json_response['concern_level'],
                    'moderation_action': json_response['moderation_action'],
                    'confidence': json_response['confidence'],
                    'status': 'success',
                    'safety_flag': 'N/A',
                    'raw_response': raw_response
                }
            except json.JSONDecodeError as je:
                # Print the invalid response to stderr
                print(f"Error parsing JSON for phrase '{phrase}':", file=sys.stderr)
                print(f"Raw response:", file=sys.stderr)
                print(raw_response, file=sys.stderr)
                print("Cleaned response:", file=sys.stderr)
                print(cleaned_response, file=sys.stderr)
                print(f"JSON error: {str(je)}", file=sys.stderr)
                print("-" * 80, file=sys.stderr)
                
                return {
                    'phrase': phrase,
                    'interpretation': 'Error: Invalid JSON response',
                    'concern_level': -1,
                    'moderation_action': 'Error: Invalid JSON response',
                    'confidence': -1,
                    'status': 'invalid_json',
                    'safety_flag': 'N/A',
                    'raw_response': raw_response
                }
        except Exception as e:
            error_str = str(e)
            
            # Check for rate limit
            if "429 Resource has been exhausted" in error_str:
                if not is_retry:
                    print(f"\nRate limit hit for phrase '{phrase}'. Waiting 60 seconds before retry...", file=sys.stderr)
                    time.sleep(60)
                    return await self.query_phrase(phrase, is_retry=True)
                else:
                    print("\nRate limit hit again after retry. Stopping further processing.", file=sys.stderr)
                    raise RateLimitExceeded("Rate limit exceeded even after retry")
            
            # Handle other errors (including safety blocks)
            safety_interpretation = None
            if "safety" in error_str.lower():
                if "harassment" in error_str.lower():
                    safety_interpretation = "BLOCKED: Potential harassment content"
                elif "hate" in error_str.lower():
                    safety_interpretation = "BLOCKED: Potential hate speech"
                elif "sexually explicit" in error_str.lower():
                    safety_interpretation = "BLOCKED: Potential sexually explicit content"
                elif "dangerous" in error_str.lower():
                    safety_interpretation = "BLOCKED: Potential dangerous content"
                else:
                    safety_interpretation = "BLOCKED: General safety concerns"
            
            return {
                'phrase': phrase,
                'interpretation': error_str,
                'concern_level': 1.0 if safety_interpretation else -1,
                'moderation_action': safety_interpretation or 'Error occurred',
                'confidence': 1.0 if safety_interpretation else 0.0,
                'status': 'blocked' if safety_interpretation else 'error',
                'safety_flag': safety_interpretation or 'N/A',
                'raw_response': error_str
            }
        except Exception as e:
            error_str = str(e)
            
            # Interpret safety-related errors
            safety_interpretation = None
            if "safety" in error_str.lower():
                if "harassment" in error_str.lower():
                    safety_interpretation = "BLOCKED: Potential harassment content"
                elif "hate" in error_str.lower():
                    safety_interpretation = "BLOCKED: Potential hate speech"
                elif "sexually explicit" in error_str.lower():
                    safety_interpretation = "BLOCKED: Potential sexually explicit content"
                elif "dangerous" in error_str.lower():
                    safety_interpretation = "BLOCKED: Potential dangerous content"
                else:
                    safety_interpretation = "BLOCKED: General safety concerns"
            
            return {
                'phrase': phrase,
                'interpretation': error_str,
                'concern_level': 1.0 if safety_interpretation else -1,
                'moderation_action': safety_interpretation or 'Error occurred',
                'confidence': 1.0 if safety_interpretation else 0.0,
                'status': 'blocked' if safety_interpretation else 'error',
                'safety_flag': safety_interpretation or 'N/A'
            }

def read_phrases(filepath: str) -> List[str]:
    """Read phrases from a file, one per line"""
    with open(filepath, 'r') as f:
        return [line.strip() for line in f if line.strip()]

def save_results(results: List[Dict], output_file: str):
    """Save results to a TSV file"""
    # Clean and escape text fields to handle tabs and newlines
    def clean_text(text: str) -> str:
        return text.replace('\n', ' ').replace('\t', ' ').strip()
    
    with open(output_file, 'w', encoding='utf-8') as f:
        # Write header
        f.write('phrase\tinterpretation\tconcern_level\tmoderation_action\tconfidence\tstatus\tsafety_flag\traw_response\n')
        
        # Write data rows
        for result in results:
            row = [
                clean_text(result['phrase']),
                clean_text(result['interpretation']),
                str(result['concern_level']),
                clean_text(result['moderation_action']),
                str(result['confidence']),
                clean_text(result['status']),
                clean_text(result['safety_flag']),
                clean_text(result.get('raw_response', 'N/A'))
            ]
            f.write('\t'.join(row) + '\n')

async def main():
    # Configuration
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key:
        raise ValueError("Please set GEMINI_API_KEY environment variable")
    
    input_file = "phrases.txt"  # Change this to your input file path
    output_file = "gemini_responses.tsv"
    
    # Initialize queryer
    queryer = GeminiQueryer(api_key)
    
    # Read phrases
    phrases = read_phrases(input_file)
    print(f"Loaded {len(phrases)} phrases")
    
    # Process phrases
    try:
        results = []
        for i, phrase in enumerate(phrases, 1):
            print(f"Processing phrase {i}/{len(phrases)}: {phrase}")
            
            try:
                result = await queryer.query_phrase(phrase)
                results.append(result)
                
                # Add delay to respect rate limits
                time.sleep(1)
            except RateLimitExceeded:
                print(f"\nStopping processing at phrase {i}/{len(phrases)} due to rate limit", file=sys.stderr)
                break
    except Exception as e:
        print(f"An error occurred: {str(e)}", file=sys.stderr)
        if not results:  # If no results were processed, re-raise the error
            raise
    finally:
        # Save any results we got
        if results:
            save_results(results, output_file)
            print(f"Results saved to {output_file}")
            
            # Report completion status
            if len(results) < len(phrases):
                print(f"\nPartially completed: processed {len(results)}/{len(phrases)} phrases before stopping", file=sys.stderr)
            else:
                print("\nSuccessfully processed all phrases")

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
