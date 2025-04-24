import os
import json
import requests
from tqdm import tqdm  # 用于显示进度条


# 配置参数
API_URL = "http://localhost:8000/v1/chat/completions"
SPLIT_FILES_DIR = "split_markdown_docs"
OUTPUT_DIR = "qa_dataset"
NUM_QUESTIONS_PER_FILE = 3
MODEL_NAME = "Qwen/QwQ-32B-AWQ"
ALL_QA_OUTPUT_FILE = "all_qa_pairs.json"  # 所有问答对的输出文件名


def generate_qa_pairs(text):
    """
    根据输入文本通过 API 调用生成问答对
    :param text: 输入的文本内容
    :return: 生成的问答对列表
    """
    user_content = """
- Role: Document Content Analysis Expert and Q&A Pair Generation Engineer
- Background: The user needs to conduct an in-depth analysis of document fragments.
- Profile: You are an expert in natural language processing and document analysis, with a strong ability to extract key information from texts and transform it into structured Q&A pairs. Your understanding of document content is exceptional, allowing you to efficiently generate accurate and relevant Q&A pairs.
- Skills: You possess capabilities in text analysis, information extraction, natural language generation, and data structuring. You can quickly grasp the core content of document fragments and generate Q&A pairs that meet the requirements.
- Goals: To generate accurate Q&A pairs based on document fragments, output in JSON format, with each Q&A pair containing 'question' and 'answer' fields.
- Constrains: The generated Q&A pairs should accurately reflect the core content of the document fragments, avoiding irrelevant information and ensuring the logic and relevance between questions and answers.
- OutputFormat: JSON format, with each Q&A pair containing 'question' and 'answer' fields.
- Workflow:
  1. Carefully read the document fragment and extract key information and main points.
  2. Based on the extracted information, generate relevant questions, ensuring that the questions are targeted and clear.
  3. Generate accurate answers for each question, ensuring that the answers are closely related to the questions and accurately reflect the document content.
- Examples:
  - Example 1:
    Document fragment: The capital of France is Paris.
    Output:
    ```json
    [
      {
        "question": "What is the capital of France?",
        "answer": "Paris"
      }
    ]
    ```
  - Example 2:
    Document fragment: Water boils at 100 degrees Celsius at sea level.
    Output:
    ```json
    [
      {
        "question": "At what temperature does water boil at sea level?",
        "answer": "100 degrees Celsius"
      }
    ]
    ```
  - Example 3:
    Document fragment: Albert Einstein was born in Ulm, Germany, in 1879.
    Output:
    ```json
    [
      {
        "question": "Where was Albert Einstein born?",
        "answer": "Ulm, Germany"
      },
      {
        "question": "When was Albert Einstein born?",
        "answer": "1879"
      }
    ]
    ```
- Initialization: In the first conversation, please directly output the following: As a document content analysis expert and Q&A pair generation engineer, I will help you convert document fragments into structured Q&A pairs in JSON format. Please provide the document fragment you want to process.
""" + f"\nDocument fragment: {text}"

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

    all_qa_pairs = []  # 存储所有问答对

    # 获取文件列表并使用 tqdm 显示进度条
    file_list = [f for f in os.listdir(SPLIT_FILES_DIR) if f.endswith('.md')]
    for filename in tqdm(file_list, desc="Processing Files", unit="file"):
        file_path = os.path.join(SPLIT_FILES_DIR, filename)
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()

        # 生成问答对
        qa_pairs = generate_qa_pairs(content)
        all_qa_pairs.extend(qa_pairs)

        # 将问答对保存到单独的文件中
        output_filename = os.path.splitext(filename)[0] + '_qa.json'
        output_path = os.path.join(OUTPUT_DIR, output_filename)
        with open(output_path, 'w', encoding='utf-8') as output_file:
            json.dump(qa_pairs, output_file, ensure_ascii=False, indent=4)
        print(f"为 {filename} 生成的问答对已保存到 {output_path}")

    # 将所有问答对保存到一个文件中
    all_qa_output_path = os.path.join(OUTPUT_DIR, ALL_QA_OUTPUT_FILE)
    with open(all_qa_output_path, 'w', encoding='utf-8') as all_qa_file:
        json.dump(all_qa_pairs, all_qa_file, ensure_ascii=False, indent=4)
    print(f"所有问答对已保存到 {all_qa_output_path}")


if __name__ == "__main__":
    process_files()
