import os
import json
import requests

# 配置参数
API_URL = "http://localhost:8000/v1/chat/completions"
SPLIT_FILES_DIR = "split_markdown_docs"
OUTPUT_DIR = "qa_dataset"
NUM_QUESTIONS_PER_FILE = 3
MODEL_NAME = "Qwen/QwQ-32B-AWQ"


def generate_qa_pairs(text):
    """
    根据输入文本通过 API 调用生成问答对
    :param text: 输入的文本内容
    :return: 生成的问答对列表
    """
    messages = [
        {
            "role": "user",
            "content": f"请根据以下文本生成{NUM_QUESTIONS_PER_FILE}个问答对，以 JSON 数组形式输出，每个元素包含 'question' 和 'answer' 字段：\n{text}"
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
        try:
            qa_pairs = json.loads(answer)
            return qa_pairs
        except json.JSONDecodeError:
            print(f"无法解析生成的文本为 JSON: {answer}")
            return []
    except requests.RequestException as e:
        print(f"请求 API 时出错: {e}")
        return []


def process_files():
    """
    处理切分后的文件，生成问答对并保存
    """
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    for filename in os.listdir(SPLIT_FILES_DIR):
        if filename.endswith('.md'):
            file_path = os.path.join(SPLIT_FILES_DIR, filename)
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
            qa_pairs = generate_qa_pairs(content)
            output_filename = os.path.splitext(filename)[0] + '_qa.json'
            output_path = os.path.join(OUTPUT_DIR, output_filename)
            with open(output_path, 'w', encoding='utf-8') as output_file:
                json.dump(qa_pairs, output_file, ensure_ascii=False, indent=4)
            print(f"为 {filename} 生成的问答对已保存到 {output_path}")


if __name__ == "__main__":
    process_files()
    