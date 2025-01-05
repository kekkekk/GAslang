import openai
import time
import json
import sys
import re
import os
from typing import List, Dict
from pathlib import Path

class RateLimitExceeded(Exception):
    """Exception raised when rate limit is hit even after retry"""
    pass

class OpenAIQueryer:
    def __init__(self, api_key: str):
        """Initialize OpenAI API client"""
        self.client = openai.OpenAI(api_key=api_key)
        
    def create_prompt(self, phrase: str) -> str:
        """Create the prompt template for a given phrase"""
        return f"""If you were a content moderator and you came across the term '{phrase}' how would you interpret it and how would you respond?

Return your response in this exact JSON format (replace the values but keep the format):
{{
"interpretation": "write what the term means and how it's used",
"concern_level": 0,
"moderation_action": "what action you recommend as a content moderator",
"confidence": 0.5
}}"""
    
    async def query_phrase(self, phrase: str, is_retry: bool = False) -> Dict:
        """Query OpenAI API for a single phrase and return the response"""
        try:
            prompt = self.create_prompt(phrase)
            
            # Use GPT-4 with JSON mode for reliable structured output
            response = self.client.chat.completions.create(
                model="gpt-4-turbo-preview",
                response_format={ "type": "json_object" },
                messages=[
                    {"role": "system", "content": "You are a content moderation assistant. Always respond with valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7
            )
            
            raw_response = response.choices[0].message.content
            
            # Check for quota error in response
            if any(error_msg in raw_response.lower() for error_msg in ["rate limit", "quota exceeded", "too many requests"]):
                if not is_retry:
                    print(f"\nRate limit hit for phrase '{phrase}'. Waiting 60 seconds before retry...", file=sys.stderr)
                    time.sleep(60)
                    return await self.query_phrase(phrase, is_retry=True)
                else:
                    print("\nRate limit hit again after retry. Stopping further processing.", file=sys.stderr)
                    raise RateLimitExceeded("Rate limit exceeded even after retry")
            
            # Parse JSON response
            try:
                json_response = json.loads(raw_response)
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
                print(f"Error parsing JSON for phrase '{phrase}':", file=sys.stderr)
                print(f"Raw response:", file=sys.stderr)
                print(raw_response, file=sys.stderr)
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
                
        except openai.RateLimitError:
            if not is_retry:
                print(f"\nRate limit hit for phrase '{phrase}'. Waiting 60 seconds before retry...", file=sys.stderr)
                time.sleep(60)
                return await self.query_phrase(phrase, is_retry=True)
            else:
                print("\nRate limit hit again after retry. Stopping further processing.", file=sys.stderr)
                raise RateLimitExceeded("Rate limit exceeded even after retry")
                
        except Exception as e:
            error_str = str(e)
            return {
                'phrase': phrase,
                'interpretation': error_str,
                'concern_level': -1,
                'moderation_action': 'Error occurred',
                'confidence': 0.0,
                'status': 'error',
                'safety_flag': 'N/A',
                'raw_response': error_str
            }

def read_phrases(filepath: str) -> List[str]:
    """Read phrases from a file, one per line"""
    with open(filepath, 'r') as f:
        return [line.strip() for line in f if line.strip()]

def save_results(results: List[Dict], output_file: str):
    """Save results to a TSV file"""
    def clean_text(text: str) -> str:
        return str(text).replace('\n', ' ').replace('\t', ' ').strip()
    
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
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        raise ValueError("Please set OPENAI_API_KEY environment variable")
    
    input_file = "phrases.txt"  # Change this to your input file path
    output_file = "openai_responses.tsv"
    
    # Initialize queryer
    queryer = OpenAIQueryer(api_key)
    
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
