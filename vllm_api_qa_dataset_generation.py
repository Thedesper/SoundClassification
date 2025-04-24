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

def generate_qa_pairs(text):
    """
    Generate QA pairs via API call with strict JSON output
    """
    # English Prompt forcing pure JSON output
    user_content = f"""
    Instructions:
    1. Act as a document analysis expert and generate Q&A pairs from the provided text.
    2. Output ONLY the JSON-formatted Q&A pairs - no explanations or extra text.
    3. Each entry must contain "question" and "answer" fields.
    4. If no valid pairs can be generated, return an empty list [].
    5. Example:
       - Input: "The capital of France is Paris."
       - Output: [{"question": "What is the capital of France?", "answer": "Paris"}]
    Document fragment: {text}
    """

    messages = [
        {"role": "user", "content": user_content}
    ]
    payload = {
        "model": MODEL_NAME,
        "messages": messages,
        "temperature": 0.0  # Force deterministic output
    }
    
    try:
        response = requests.post(API_URL, json=payload)
        response.raise_for_status()
        result = response.json()
        answer = result.get("choices", [{}])[0].get("message", {}).get("content", "")
        
        # 强制提取有效JSON部分
        if not answer:
            print("API returned empty content")
            return []
        
        # 确保输出严格为JSON格式
        if not answer.strip().startswith("["):
            # 如果有非JSON前缀，尝试截取
            start = answer.find("[")
            end = answer.rfind("]")
            answer = answer[start:end+1] if start != -1 and end != -1 else "[]"
        
        try:
            qa_pairs = json.loads(answer)
            # 验证每个条目结构
            for pair in qa_pairs:
                if not isinstance(pair, dict) or "question" not in pair or "answer" not in pair:
                    print(f"Invalid QA pair structure: {pair}")
                    return []
            return qa_pairs
        except json.JSONDecodeError as e:
            print(f"JSON parse error: {e}")
            print(f"Raw API response: {answer}")
            return []
    except requests.RequestException as e:
        print(f"API request failed: {e}")
        return []

def save_to_file(data, file_path):
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        print(f"Saved to {file_path}")
    except Exception as e:
        print(f"Save failed: {e}")

def process_files():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    all_qa = []
    
    file_list = [f for f in os.listdir(SPLIT_FILES_DIR) if f.endswith('.md')]
    for filename in tqdm(file_list, desc="Processing", unit="file"):
        file_path = os.path.join(SPLIT_FILES_DIR, filename)
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            qa_pairs = generate_qa_pairs(content)
            if not qa_pairs:
                print(f"Warning: No QA pairs generated for {filename}")
                continue
            all_qa.extend(qa_pairs)
            
            output_path = os.path.join(OUTPUT_DIR, f"{os.path.splitext(filename)[0]}_qa.json")
            save_to_file(qa_pairs, output_path)
        except Exception as e:
            print(f"Error processing {filename}: {e}")
    
    save_to_file(all_qa, os.path.join(OUTPUT_DIR, ALL_QA_OUTPUT_FILE))

if __name__ == "__main__":
    process_files()
