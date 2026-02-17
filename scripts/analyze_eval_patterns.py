
import json
import os

def analyze_discrepancies(file_path, lang):
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    discrepancies = []
    if isinstance(data, dict) and 'samples' in data:
        samples = data['samples']
    elif isinstance(data, dict):
        # Maybe it's a dict of samples?
        samples = list(data.values())
    else:
        samples = data
        
    for i, item in enumerate(samples):
        if not isinstance(item, dict): continue
        
        scores = item.get('scores', {})
        judge = scores.get('judge', {})
        
        rougeL = scores.get('rougeL', 0)
        faithfulness = judge.get('faithfulness', 0)
        completeness = judge.get('completeness', 0)
        
        # Pattern 1: Low ROUGE, High Judge (Highly abstract or different focus but correct)
        if rougeL < 0.15 and faithfulness >= 4 and completeness >= 4:
            discrepancies.append({
                'index': i,
                'type': 'Low ROUGE / High Judge',
                'rougeL': rougeL,
                'faithfulness': faithfulness,
                'completeness': completeness,
                'reason': str(judge.get('faithfulness_reason', ''))[:100]
            })
            
        # Pattern 2: High ROUGE, Low Judge (Heavy copying but flawed)
        if rougeL > 0.6 and (faithfulness <= 2 or completeness <= 2):
            discrepancies.append({
                'index': i,
                'type': 'High ROUGE / Low Judge',
                'rougeL': rougeL,
                'faithfulness': faithfulness,
                'completeness': completeness,
                'reason': str(judge.get('faithfulness_reason', ''))[:100]
            })

        # Pattern 3: Unable to evaluate
        reason = str(judge.get('faithfulness_reason', ''))
        if "Unable to evaluate" in reason:
            discrepancies.append({
                'index': i,
                'type': 'Unable to Evaluate',
                'reason': reason
            })

    print(f"\n--- {lang.upper()} DISCREPANCIES ---")
    for d in discrepancies:
        print(f"Sample {d.get('index')}: [{d['type']}] ROUGE: {d.get('rougeL', 'N/A')}, F: {d.get('faithfulness', 'N/A')}, C: {d.get('completeness', 'N/A')}")
        print(f"  Reason: {d['reason']}")

analyze_discrepancies('/home/marcantoniolopez/Documenti/github/projects/sm-sip/notebooks/evaluation/results/english/eval_enhanced.json', 'english')
analyze_discrepancies('/home/marcantoniolopez/Documenti/github/projects/sm-sip/notebooks/evaluation/results/italian/eval_enhanced.json', 'italian')
