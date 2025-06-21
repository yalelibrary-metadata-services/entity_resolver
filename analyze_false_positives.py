#!/usr/bin/env python3
"""
Analyze false positive patterns by examining specific records from the training data.
"""

import pandas as pd
import sys

def analyze_false_positives():
    """Examine specific false positive cases."""
    
    # Load the training data
    print("Loading training data...")
    df = pd.read_csv('data/input/training_dataset_classified.csv')
    print(f"Loaded {len(df)} records")
    
    # Sample false positive pairs from the test results
    false_positive_pairs = [
        ("10648396#Agent600-19", "10658209#Agent600-19"),  # HIGH confidence (0.92), perfect person match
        ("10667180#Agent600-20", "10696178#Agent600-20"),  # HIGH confidence (0.92), perfect person match  
        ("10721964#Agent600-19", "5610980#Agent700-23"),   # Cross Agent600/700, different ending numbers
        ("11670851#Agent700-19", "11266484#Agent700-15"),  # Both Agent700, different ending numbers
        ("10721964#Agent600-19", "11449955#Agent100-11"),  # Cross Agent600/100
        ("730544#Agent700-26", "880495#Agent700-22"),      # Both Agent700, different ending numbers
        ("583878#Agent100-12", "838486#Agent700-23"),      # Cross Agent100/700
    ]
    
    print("\n" + "="*80)
    print("ANALYZING FALSE POSITIVE PATTERNS")
    print("="*80)
    
    for i, (left_id, right_id) in enumerate(false_positive_pairs, 1):
        print(f"\n{'='*20} FALSE POSITIVE PAIR {i} {'='*20}")
        print(f"LEFT:  {left_id}")
        print(f"RIGHT: {right_id}")
        
        # Find records for each ID
        left_records = df[df['personId'] == left_id]
        right_records = df[df['personId'] == right_id]
        
        if len(left_records) == 0:
            print(f"❌ No records found for LEFT ID: {left_id}")
        else:
            left_record = left_records.iloc[0]
            print(f"\nLEFT RECORD:")
            print(f"  Identity: {left_record['identity']}")
            print(f"  Person: {left_record['person']}")
            print(f"  Title: {left_record['title']}")
            print(f"  Role: {left_record['roles']}")
            print(f"  Taxonomy: {left_record['setfit_prediction']}")
            print(f"  Subjects: {left_record['subjects']}")
            print(f"  Provision: {left_record['provision']}")
            
        if len(right_records) == 0:
            print(f"❌ No records found for RIGHT ID: {right_id}")
        else:
            right_record = right_records.iloc[0]
            print(f"\nRIGHT RECORD:")
            print(f"  Identity: {right_record['identity']}")
            print(f"  Person: {right_record['person']}")
            print(f"  Title: {right_record['title']}")
            print(f"  Role: {right_record['roles']}")
            print(f"  Taxonomy: {right_record['setfit_prediction']}")
            print(f"  Subjects: {right_record['subjects']}")
            print(f"  Provision: {right_record['provision']}")
            
        # Analysis
        if len(left_records) > 0 and len(right_records) > 0:
            print(f"\n🔍 ANALYSIS:")
            print(f"  Same person name: {left_record['person'] == right_record['person']}")
            print(f"  Same identity: {left_record['identity'] == right_record['identity']}")
            print(f"  Same taxonomy: {left_record['setfit_prediction'] == right_record['setfit_prediction']}")
            print(f"  Same role: {left_record['roles'] == right_record['roles']}")
            
            # Check if same person string but different identities
            if (left_record['person'] == right_record['person'] and 
                left_record['identity'] != right_record['identity']):
                print(f"  ⚠️  SAME PERSON NAME, DIFFERENT IDENTITIES!")
                print(f"       This explains why person_cosine=1.0 but should be FALSE")
                
            # Check for date/temporal differences
            left_dates = extract_dates_from_text(str(left_record['provision']) + " " + str(left_record['subjects']))
            right_dates = extract_dates_from_text(str(right_record['provision']) + " " + str(right_record['subjects']))
            if left_dates or right_dates:
                print(f"  Dates - Left: {left_dates}, Right: {right_dates}")
                
    print(f"\n{'='*60}")
    print("SUMMARY PATTERNS:")
    print("="*60)
    
    # Count patterns across all false positives
    agent_pattern_counts = {}
    name_identity_mismatches = 0
    
    for left_id, right_id in false_positive_pairs:
        left_records = df[df['personId'] == left_id]
        right_records = df[df['personId'] == right_id]
        
        if len(left_records) > 0 and len(right_records) > 0:
            left_record = left_records.iloc[0]
            right_record = right_records.iloc[0]
            
            # Check agent pattern
            left_agent = extract_agent_pattern(left_id)
            right_agent = extract_agent_pattern(right_id)
            pattern = f"{left_agent} vs {right_agent}"
            agent_pattern_counts[pattern] = agent_pattern_counts.get(pattern, 0) + 1
            
            # Check name/identity mismatches
            if (left_record['person'] == right_record['person'] and 
                left_record['identity'] != right_record['identity']):
                name_identity_mismatches += 1
    
    print(f"Agent ID patterns:")
    for pattern, count in sorted(agent_pattern_counts.items()):
        print(f"  {pattern}: {count} cases")
        
    print(f"\nSame name, different identity: {name_identity_mismatches} cases")
    print(f"Total analyzed: {len(false_positive_pairs)} pairs")

def extract_agent_pattern(person_id):
    """Extract the Agent pattern from personId like 'Agent600-19'."""
    if '#Agent' in person_id:
        agent_part = person_id.split('#Agent')[1]
        if '-' in agent_part:
            return f"Agent{agent_part.split('-')[0]}"
    return "Unknown"

def extract_dates_from_text(text):
    """Extract 4-digit years from text."""
    import re
    if pd.isna(text) or text == 'nan':
        return []
    dates = re.findall(r'\b(19|20)\d{2}\b', str(text))
    return [d[0] + d[1] for d in dates]

if __name__ == "__main__":
    analyze_false_positives()