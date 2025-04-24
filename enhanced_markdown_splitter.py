from langchain.text_splitter import MarkdownHeaderTextSplitter, MarkdownTextSplitter
import os
import pandas as pd

# 配置参数
SOURCE_MD_PATH = os.path.join(os.path.pardir, 'outputs', 'MinerU_parsed_20241204', '2024全球经济金融展望报告.md')
OUTPUT_DIR = "split_markdown_docs"  # 输出文件夹名称
MAX_CHUNK_LENGTH = 700  # 触发二次切分的长度阈值
CHUNK_SIZE = 500  # 二次切分的块大小
CHUNK_OVERLAP = 50  # 块重叠量

def load_markdown_document(file_path):
    """加载Markdown文档并处理编码"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()

def title_preserving_splitter(text):
    """基于标题切分并保留标题在切片内容中"""
    headers_to_split = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3")
    ]
    return MarkdownHeaderTextSplitter(headers_to_split, strip_headers=False).split_text(text)

def propagate_headers_to_chunks(original_doc, sub_chunks):
    """将原始文档的标题信息传播到二次切分的子片段"""
    if not sub_chunks: return []
    header_lines = [
        f'{"#"*level} {original_doc.metadata.get(f"Header {level}", "")}' 
        for level in range(1, 4) 
        if f"Header {level}" in original_doc.metadata
    ]
    for doc in sub_chunks[1:]:  # 首个切片已包含完整标题，后续切片补充标题前缀
        doc.page_content = '\n'.join(header_lines) + '\n' + doc.page_content
    return sub_chunks

def optimize_chunk_length(original_docs, max_length=MAX_CHUNK_LENGTH):
    """对超长切片进行二次切分并保留标题上下文"""
    splitter = MarkdownTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    optimized_docs = []
    for doc in original_docs:
        if len(doc.page_content) > max_length:
            sub_chunks = splitter.split_documents([doc])
            optimized_docs.extend(propagate_headers_to_chunks(doc, sub_chunks))
        else:
            optimized_docs.append(doc)
    return optimized_docs

def save_chunks_to_disk(chunks, output_dir=OUTPUT_DIR):
    """将切分后的文档保存到指定文件夹"""
    os.makedirs(output_dir, exist_ok=True)  # 创建输出目录
    for idx, doc in enumerate(chunks, 1):
        # 生成带标题的文件名（使用一级标题，若无则用默认命名）
        title = doc.metadata.get("Header 1", f"chunk_{idx}").replace("/", "_")  # 处理非法字符
        file_name = f"{idx:03d}_{title[:30]}.md"  # 限制文件名长度防止过长
        file_path = os.path.join(output_dir, file_name)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(doc.page_content)
        
        print(f"已保存切片 {idx}: {file_path}")

# 完整处理流程
if __name__ == "__main__":
    # 1. 加载原始文档
    markdown_content = load_markdown_document(SOURCE_MD_PATH)
    
    # 2. 首次切分（保留标题）
    first_split_docs = title_preserving_splitter(markdown_content)
    
    # 3. 二次切分优化长度
    final_optimized_docs = optimize_chunk_length(first_split_docs)
    
    # 4. 保存到磁盘
    save_chunks_to_disk(final_optimized_docs)
    
    # 5. 输出统计信息
    final_length_stats = pd.Series([len(doc.page_content) for doc in final_optimized_docs]).describe()
    print("\n最终切片长度统计：")
    print(final_length_stats)
  