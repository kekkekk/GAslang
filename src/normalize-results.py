import pandas as pd
import json
from datetime import datetime

def normalize_response(row):
    """Normalize response format to latest version"""
    try:
        # If response is already a string representation of a dict, parse it
        if isinstance(row['response'], str):
            try:
                response = json.loads(row['response'])
            except json.JSONDecodeError:
                # Handle case where response contains error message
                if 'Error:' in str(row['response']):
                    response = {
                        "interpretation": str(row['response']),
                        "concern_level": 5 if "safety concerns" in str(row['response']).lower() else None,
                        "moderation_action": "Content blocked" if "blocked" in str(row['response']).lower() else "Error processing content",
                        "confidence": 1.0 if "safety concerns" in str(row['response']).lower() else 0,
                        "safety_analysis": {}
                    }
                else:
                    response = {
                        "interpretation": str(row['response']),
                        "concern_level": None,
                        "moderation_action": None,
                        "confidence": 0,
                        "safety_analysis": {}
                    }
        else:
            response = row['response']

        # Ensure all required fields exist
        if 'interpretation' not in response:
            response['interpretation'] = "No interpretation provided"
        if 'concern_level' not in response:
            response['concern_level'] = None
        if 'moderation_action' not in response:
            response['moderation_action'] = None
        if 'confidence' not in response:
            response['confidence'] = 0
        if 'safety_analysis' not in response:
            response['safety_analysis'] = {}

        # Parse safety information from error messages
        if isinstance(row.get('error'), str) and 'safety' in row['error'].lower():
            try:
                # Extract safety ratings from error message
                if 'safety_ratings are:' in row['error']:
                    safety_part = row['error'].split('safety_ratings are:')[1]
                    categories = safety_part.strip('[] ').split(', ')
                    
                    safety_analysis = {}
                    for cat in categories:
                        if 'category:' in cat and 'probability:' in cat:
                            cat_parts = cat.split('\n')
                            category = cat_parts[0].split(':')[1].strip()
                            prob = cat_parts[1].split(':')[1].strip()
                            
                            # Convert probability to score
                            prob_map = {
                                'NEGLIGIBLE': 0,
                                'LOW': 0.25,
                                'MEDIUM': 0.5,
                                'HIGH': 0.75,
                                'VERY_HIGH': 1.0
                            }
                            prob_score = prob_map.get(prob, 0)
                            
                            safety_analysis[category] = {
                                'probability': prob,
                                'score': prob_score
                            }
                    
                    response['safety_analysis'] = safety_analysis
                    
                    # Update other fields based on safety analysis
                    max_prob_score = max((v['score'] for v in safety_analysis.values()), default=0)
                    response['confidence'] = max_prob_score
                    response['concern_level'] = min(5, round(max_prob_score * 5))
                    response['moderation_action'] = "Content requires review/modification" if max_prob_score < 0.75 else "Content should be blocked"
                    response['interpretation'] = f"Content flagged for safety concerns: {', '.join(f'{k}: {v['probability']}' for k,v in safety_analysis.items() if v['score'] >= 0.5)}"
            except Exception as e:
                print(f"Error parsing safety info: {str(e)}")

        return json.dumps(response)
    except Exception as e:
        print(f"Error normalizing row: {str(e)}")
        return json.dumps({
            "interpretation": "Error normalizing response",
            "concern_level": None,
            "moderation_action": None,
            "confidence": 0,
            "safety_analysis": {}
        })

def main():
    # Read the CSV file
    df = pd.read_csv('GeminiEvaluator_results_155.csv')
    
    # Normalize each response
    df['response'] = df.apply(normalize_response, axis=1)
    
    # Save normalized results
    output_file = 'GeminiEvaluator_results_normalized.csv'
    df.to_csv(output_file, index=False)
    print(f"Normalized results saved to {output_file}")
    
    # Print some statistics
    total_rows = len(df)
    error_rows = df['error'].notna().sum()
    print(f"\nProcessed {total_rows} rows")
    print(f"Found {error_rows} rows with errors")
    
    # Sample a few responses to verify format
    print("\nSample of normalized responses:")
    sample_responses = df.sample(min(3, len(df)))
    for _, row in sample_responses.iterrows():
        print(f"\nPhrase: {row['phrase']}")
        try:
            resp = json.loads(row['response'])
            print(f"Concern Level: {resp['concern_level']}")
            print(f"Confidence: {resp['confidence']}")
            if resp['safety_analysis']:
                print("Safety Analysis:", json.dumps(resp['safety_analysis'], indent=2))
        except:
            print("Error parsing response")

if __name__ == "__main__":
    main()
