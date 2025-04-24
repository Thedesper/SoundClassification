import os
import json
import requests
from tqdm import tqdm  # 用于显示进度条

# 配置参数
API_URL = "http://localhost:8000/v1/chat/completions"
SPLIT_FILES_DIR = "split_markdown_docs"
OUTPUT_DIR = "qa_dataset"
MODEL_NAME = "Qwen/QwQ-32B-AWQ"
ALL_QA_OUTPUT_FILE = "all_qa_pairs.json"

def generate_qa_pairs(text):
    """
    根据输入文本通过 API 调用生成问答对
    :param text: 输入的文本内容
    :return: 生成的问答对列表
    """
    # 简化后的 Prompt，强制模型仅输出 JSON 格式的问答对
    user_content = f"""
    指令：
    1. 你是一个文档分析专家，根据提供的文本生成问答对。
    2. 仅输出 JSON 格式的问答对，不要任何解释或额外内容。
    3. 每个问答对必须包含 "question" 和 "answer" 字段。
    4. 如果无法生成有效问答对，返回空列表 []。
    5. 示例：
       - 输入："The capital of France is Paris."
       - 输出：[{{"question": "What is the capital of France?", "answer": "Paris"}}]
    文档片段：{text}
    """

    messages = [
        {
            "role": "user",
            "content": user_content
        }
    ]
    payload = {
        "model": MODEL_NAME,
        "messages": messages
    }
    try:
        response = requests.post(API_URL, json=payload)
        response.raise_for_status()
        result = response.json()
        answer = result.get("choices", [{}])[0].get("message", {}).get("content", "")
        
        # 强制仅保留 JSON 内容，过滤其他文本
        if not answer:
            print("API 返回内容为空")
            return []
        
        # 尝试解析 JSON，仅保留有效部分
        try:
            qa_pairs = json.loads(answer)
            # 验证 JSON 结构是否符合要求
            for pair in qa_pairs:
                if not isinstance(pair, dict) or "question" not in pair or "answer" not in pair:
                    print(f"无效的问答对结构: {pair}")
                    return []
            return qa_pairs
        except json.JSONDecodeError as e:
            print(f"JSON 解析失败：{e}")
            print(f"原始 API 返回内容：{answer}")
            return []
    except requests.RequestException as e:
        print(f"请求 API 时出错: {e}")
        return []

def save_to_file(data, file_path):
    try:
        with open(file_path, 'w', encoding='utf-8') as file:
            json.dump(data, file, ensure_ascii=False, indent=4)
        print(f"成功保存到 {file_path}")
    except Exception as e:
        print(f"保存文件失败: {e}")

def process_files():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    all_qa_pairs = []

    file_list = [f for f in os.listdir(SPLIT_FILES_DIR) if f.endswith('.md')]
    for filename in tqdm(file_list, desc="Processing", unit="file"):
        file_path = os.path.join(SPLIT_FILES_DIR, filename)
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            qa_pairs = generate_qa_pairs(content)
            if not qa_pairs:
                print(f"警告：{filename} 生成空问答对，跳过保存")
                continue
            all_qa_pairs.extend(qa_pairs)
            
            output_path = os.path.join(OUTPUT_DIR, f"{os.path.splitext(filename)[0]}_qa.json")
            save_to_file(qa_pairs, output_path)
        except Exception as e:
            print(f"处理 {filename} 失败: {e}")
    
    save_to_file(all_qa_pairs, os.path.join(OUTPUT_DIR, ALL_QA_OUTPUT_FILE))

if __name__ == "__main__":
    process_files()
