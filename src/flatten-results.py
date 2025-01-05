import pandas as pd
import json

def clean_value(value):
    """Clean and format values for TSV output"""
    if value is None:
        return ""
    if isinstance(value, (int, float)):
        return value
    # Replace tabs and newlines with spaces
    return str(value).strip().replace('\t', ' ').replace('\n', ' ')

def flatten_interpretation(interp_text):
    """Convert interpretation text to structured fields if possible"""
    if isinstance(interp_text, str) and interp_text.startswith("Content flagged for safety concerns:"):
        return {
            'type': 'safety_flag',
            'flags': interp_text.split(':', 1)[1].strip()
        }
    return {
        'type': 'standard',
        'text': str(interp_text)
    }

def flatten_response(row):
    """Extract fields from JSON response into separate columns"""
    try:
        # Parse response JSON
        response = json.loads(row['response'])
        flattened = {}
        
        # Extract interpretation
        interpretation_raw = response.get('interpretation', '')
        interpretation_parsed = flatten_interpretation(interpretation_raw)
        flattened['interpretation_type'] = clean_value(interpretation_parsed['type'])
        flattened['interpretation_flags'] = clean_value(interpretation_parsed.get('flags', ''))
        flattened['interpretation_text'] = clean_value(interpretation_parsed.get('text', ''))
        
        # Extract main fields with cleaning
        flattened['concern_level'] = clean_value(response.get('concern_level', ''))
        flattened['moderation_action'] = clean_value(response.get('moderation_action', ''))
        flattened['confidence'] = clean_value(response.get('confidence', 0))
        
        # Extract safety analysis
        safety = response.get('safety_analysis', {})
        categories = [
            'HARM_CATEGORY_SEXUALLY_EXPLICIT',
            'HARM_CATEGORY_HATE_SPEECH',
            'HARM_CATEGORY_HARASSMENT',
            'HARM_CATEGORY_DANGEROUS_CONTENT'
        ]
        
        for category in categories:
            if category in safety:
                flattened[f'{category}_prob'] = clean_value(safety[category].get('probability', 'NEGLIGIBLE'))
                flattened[f'{category}_score'] = clean_value(safety[category].get('score', 0))
            else:
                flattened[f'{category}_prob'] = 'NEGLIGIBLE'
                flattened[f'{category}_score'] = 0
        
        # Copy original columns
        flattened['phrase'] = row['phrase']
        flattened['model'] = row['model']
        flattened['timestamp'] = row['timestamp']
        if 'error' in row:
            flattened['error'] = row['error']
            
        return pd.Series(flattened)
        
    except Exception as e:
        print(f"Error processing row for phrase '{row.get('phrase', 'unknown')}': {str(e)}")
        return pd.Series({
            'phrase': row.get('phrase', 'unknown'),
            'model': row.get('model', 'unknown'),
            'timestamp': row.get('timestamp', ''),
            'interpretation_type': 'error',
            'interpretation_flags': '',
            'interpretation_text': f'Error: {str(e)}',
            'concern_level': '',
            'moderation_action': '',
            'confidence': 0,
            **{f'HARM_CATEGORY_{cat}_prob': 'NEGLIGIBLE' for cat in ['SEXUALLY_EXPLICIT', 'HATE_SPEECH', 'HARASSMENT', 'DANGEROUS_CONTENT']},
            **{f'HARM_CATEGORY_{cat}_score': 0 for cat in ['SEXUALLY_EXPLICIT', 'HATE_SPEECH', 'HARASSMENT', 'DANGEROUS_CONTENT']},
            'error': str(e)
        })

def main():
    # Read the normalized CSV file
    print("Reading input file...")
    df = pd.read_csv('GeminiEvaluator_results_normalized.csv')
    
    # Apply flattening to each row
    print("Flattening responses...")
    df_flattened = df.apply(flatten_response, axis=1)
    
    # Define column order
    column_order = [
        'phrase',
        'model',
        'timestamp',
        'interpretation_type',
        'interpretation_flags',
        'interpretation_text',
        'concern_level',
        'moderation_action',
        'confidence'
    ]
    
    # Add safety category columns
    for category in ['SEXUALLY_EXPLICIT', 'HATE_SPEECH', 'HARASSMENT', 'DANGEROUS_CONTENT']:
        column_order.extend([
            f'HARM_CATEGORY_{category}_prob',
            f'HARM_CATEGORY_{category}_score'
        ])
    
    # Add error column if it exists
    if 'error' in df_flattened.columns:
        column_order.append('error')
    
    # Reorder columns and handle any missing columns
    existing_columns = [col for col in column_order if col in df_flattened.columns]
    df_flattened = df_flattened[existing_columns]
    
    # Save flattened results
    output_file = 'GeminiEvaluator_results_flattened_detailed.tsv'
    df_flattened.to_csv(output_file, index=False, sep='\t', escapechar='\\')
    print(f"\nFlattened results saved to {output_file}")
    
    # Print statistics and verify data
    print(f"\nProcessed {len(df_flattened)} rows")
    
    # Verify the output is readable
    print("\nVerifying output file...")
    try:
        test_df = pd.read_csv(output_file, sep='\t')
        print(f"Successfully verified TSV file with {len(test_df)} rows and {len(test_df.columns)} columns")
        # Verify critical columns
        for col in ['concern_level', 'moderation_action']:
            if col in test_df.columns:
                non_empty = test_df[col].notna().sum()
                print(f"- {col}: {non_empty} non-empty values")
                if non_empty > 0:
                    print(f"  Sample value: {test_df[col].iloc[0]}")
    except Exception as e:
        print(f"Error verifying output file: {str(e)}")
    print("\nColumn value counts:")
    for col in ['concern_level', 'moderation_action']:
        if col in df_flattened.columns:
            print(f"\n{col} values:")
            print(df_flattened[col].value_counts().head())
    
    # Verify all columns are present
    print("\nColumns in output:")
    for col in df_flattened.columns:
        non_empty = df_flattened[col].notna().sum()
        print(f"- {col}: {non_empty} non-empty values")

if __name__ == "__main__":
    main()
