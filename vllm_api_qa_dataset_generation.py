import os
import json
import requests
from tqdm import tqdm

# Configuration
API_URL = "http://localhost:8000/v1/chat/completions"
SPLIT_FILES_DIR = "split_markdown_docs"
OUTPUT_DIR = "qa_dataset"
MODEL_NAME = "Qwen/QwQ-32B-AWQ"
ALL_QA_OUTPUT_FILE = "all_qa_pairs.json"

def generate_qa_pairs(text, num_pairs=10):
    """
    Generate QA pairs via API call with strict JSON output validation
    """
    structured_prompt = f"""Generate exactly {num_pairs} high-quality question-answer pairs from the following text.
    
Requirements:
1. Output STRICT JSON format with "qa_pairs" array
2. Each object must contain "question" and "answer" fields
3. Answers must be complete sentences
4. Avoid hypothetical questions
5. No markdown formatting

Example Format:
{{
  "qa_pairs": [
    {{
      "question": "What is the powerhouse of the cell?",
      "answer": "The mitochondria are known as the powerhouse of the cell."
    }}
  ]
}}

Input Text:
{text}

Output MUST be ONLY valid JSON:"""

    messages = [{"role": "user", "content": structured_prompt}]
    payload = {
        "model": MODEL_NAME,
        "messages": messages,
        "temperature": 0.3,  # Slightly relaxed for better quality
        "response_format": {"type": "json_object"}  # Enforce JSON mode
    }
    
    try:
        response = requests.post(API_URL, json=payload, timeout=30)
        response.raise_for_status()
        result = response.json()
        raw_content = result.get("choices", [{}])[0].get("message", {}).get("content", "")
        
        # JSON Sanitization Pipeline
        sanitized = raw_content.strip()
        if not sanitized:
            return []
        
        # Attempt to fix common formatting issues
        for fix in ['```json', '```']:
            if sanitized.startswith(fix):
                sanitized = sanitized[len(fix):].strip()
        
        try:
            # Primary JSON parsing
            parsed = json.loads(sanitized)
            if not isinstance(parsed, dict):
                raise ValueError("Top-level structure must be a dictionary")
            
            qa_list = parsed.get("qa_pairs", [])
            if not isinstance(qa_list, list):
                raise ValueError("qa_pairs must be an array")
            
            # Validate each pair
            valid_pairs = []
            for pair in qa_list:
                if isinstance(pair, dict) and "question" in pair and "answer" in pair:
                    valid_pairs.append({
                        "question": str(pair["question"]).strip(),
                        "answer": str(pair["answer"]).strip()
                    })
            return valid_pairs
            
        except (json.JSONDecodeError, ValueError) as e:
            # Fallback JSON extraction
            print(f"JSON Error: {str(e)} | Attempting recovery...")
            start = sanitized.find('{')
            end = sanitized.rfind('}') + 1
            if start != -1 and end != -1:
                try:
                    recovered = json.loads(sanitized[start:end])
                    return recovered.get("qa_pairs", [])
                except:
                    pass
            return []
            
    except requests.RequestException as e:
        print(f"API Error: {str(e)}")
        return []
    except Exception as e:
        print(f"Unexpected Error: {str(e)}")
        return []

def save_to_file(data, file_path):
    """Atomic write with backup preservation"""
    try:
        temp_path = f"{file_path}.tmp"
        with open(temp_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        os.replace(temp_path, file_path)
        print(f"Saved {len(data)} items to {file_path}")
    except Exception as e:
        print(f"Save Error: {str(e)}")

def process_files():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    master_dataset = []
    
    md_files = [f for f in os.listdir(SPLIT_FILES_DIR) 
                if f.endswith('.md') and os.path.isfile(os.path.join(SPLIT_FILES_DIR, f))]
    
    for filename in tqdm(sorted(md_files), desc="Processing Documents"):
        file_path = os.path.join(SPLIT_FILES_DIR, filename)
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                
            if not content:
                print(f"Empty file: {filename}")
                continue
                
            qa_data = generate_qa_pairs(content)
            if not qa_data:
                print(f"No valid QA pairs in {filename}")
                continue
                
            # Save individual file QA
            output_file = os.path.join(OUTPUT_DIR, f"{os.path.splitext(filename)[0]}_qa.json")
            save_to_file({"source": filename, "qa_pairs": qa_data}, output_file)
            
            # Aggregate for master file
            master_dataset.extend(qa_data)
            
        except Exception as e:
            print(f"Processing Error [{filename}]: {str(e)}")
            continue
    
    # Save consolidated dataset
    if master_dataset:
        save_to_file(master_dataset, os.path.join(OUTPUT_DIR, ALL_QA_OUTPUT_FILE))
    else:
        print("Warning: No QA pairs generated from any files")

if __name__ == "__main__":
    process_files()
