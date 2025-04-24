import os
import json
from vllm import LLM, SamplingParams


# 配置参数
LOCAL_MODEL_PATH = "your_local_model_path"  # 替换为你的本地模型路径
SPLIT_FILES_DIR = "split_markdown_docs"  # 切分后文件所在的文件夹
OUTPUT_DIR = "qa_pairs_output"  # 保存问答对的文件夹
NUM_QUESTIONS_PER_FILE = 3  # 每个文件生成的问答对数量
SAMPLING_PARAMS = SamplingParams(
    temperature=0.7,
    top_p=0.9,
    max_tokens=200
)


def generate_qa_pairs(text):
    """
    根据输入文本生成问答对
    :param text: 输入的文本内容
    :return: 生成的问答对列表
    """
    llm = LLM(model=LOCAL_MODEL_PATH)
    prompt = f"请根据以下文本生成{NUM_QUESTIONS_PER_FILE}个问答对，以JSON数组形式输出，每个元素包含'question'和'answer'字段：\n{text}"
    outputs = llm.generate(prompts=[prompt], sampling_params=SAMPLING_PARAMS)
    generated_text = outputs[0].outputs[0].text
    try:
        qa_pairs = json.loads(generated_text)
        return qa_pairs
    except json.JSONDecodeError:
        print(f"无法解析生成的文本为JSON: {generated_text}")
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
    