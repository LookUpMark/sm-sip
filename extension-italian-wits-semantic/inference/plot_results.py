import os
import json
import glob
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from datetime import datetime

def parse_results(root_dir):
    data = []
    
    # Find all json files starting with results_
    pattern = os.path.join(root_dir, "**", "results_*.json")
    files = glob.glob(pattern, recursive=True)
    
    for f in files:
        try:
            with open(f, 'r') as file:
                content = json.load(file)
                
            run_info = content.get('run_info', {})
            metrics = content.get('metrics', {})
            
            # Extract info
            model_id = run_info.get('sigext_model', 'unknown')
            # Simplify model name (e.g., "LookUpMark/sigext-wits-it-10k-060t" -> "10k-060t")
            if '/' in model_id:
                model_short = model_id.split('/')[-1].replace('sigext-wits-it-', '')
            else:
                model_short = model_id
                
            inference_type = run_info.get('inference_type', 'unknown')
            
            # Determine quantization from filename
            filename = os.path.basename(f)
            if '8bit' in filename:
                quantization = '8bit'
            elif '4bit' in filename:
                quantization = '4bit'
            else:
                # Fallback or assumption
                quantization = '4bit' # Assuming default is 4bit based on file lists seen
            
            # Metrics
            bert_score = metrics.get('bert_score', {}).get('mean', 0)
            rouge1 = metrics.get('rouge1', {}).get('mean', 0)
            kir = metrics.get('kir', {}).get('mean', 0)
            
            entry = {
                'Model': model_short,
                'Inference Type': inference_type,
                'Quantization': quantization,
                'BERTScore': bert_score,
                'ROUGE-1': rouge1,
                'KIR': kir,
                'File': filename
            }
            data.append(entry)
        except Exception as e:
            print(f"Error parsing {f}: {e}")
            
    return pd.DataFrame(data)

def plot_comparison(df, output_dir):
    if df.empty:
        print("No data found.")
        return

    # Set style
    sns.set_theme(style="whitegrid")
    
    # Metrics to plot
    metrics = ['BERTScore', 'ROUGE-1', 'KIR']
    
    # Create plots
    for metric in metrics:
        plt.figure(figsize=(12, 6))
        
        # We want to compare Models, grouped by Inference Type and Quantization
        df['Config'] = df['Inference Type'] + " - " + df['Quantization']
        
        g = sns.barplot(
            data=df, 
            x='Model', 
            y=metric, 
            hue='Config',
            palette="viridis"
        )
        
        plt.title(f'Comparison of {metric} across Models and Configurations')
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"comparison_{metric}_{timestamp}.png"
        save_path = os.path.join(output_dir, filename)
        plt.savefig(save_path)
        print(f"Saved plot to {save_path}")
        plt.close()

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    df = parse_results(current_dir)
    
    print("Aggregated Data:")
    print(df)
    
    if not df.empty:
        # Save aggregated CSV
        df.to_csv(os.path.join(current_dir, "aggregated_results.csv"), index=False)
        plot_comparison(df, current_dir)
