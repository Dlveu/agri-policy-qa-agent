"""
build_faiss_rag.py
------------------
功能：
- 遍历 policy_docs/ 下的 HTML 文件
- 智能清洗 HTML（过滤导航/广告/无关标签 + 非中文字符 + 垃圾关键词）
- 精准提取政策正文
- 按【中文句子】切分（稳定、适合政策）
- Embedding 后写入本地 FAISS 向量库
- 支持【增量更新】（基于文件修改时间）
- 新增：切分文本预览 + FAISS 数据导出功能
- 修复：过滤俄语字符、12345接诉即办等垃圾内容
"""

import os
import json
import re
from glob import glob
from bs4 import BeautifulSoup
import dotenv

from langchain_community.vectorstores.faiss import FAISS
from langchain_community.embeddings.openai import OpenAIEmbeddings

# =========================
# 环境变量
# =========================
dotenv.load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["OPENAI_BASE_URL"] = os.getenv("OPENAI_BASE_URL")

# =========================
# 配置
# =========================
RAG_FOLDER = "policy_docs"          # HTML 政策文件目录
FAISS_INDEX_DIR = "faiss_index"     # FAISS 向量库目录
METADATA_FILE = os.path.join(FAISS_INDEX_DIR, "metadata.json")
EXPORT_FILE = os.path.join(FAISS_INDEX_DIR, "faiss_exported_data.json")  # 导出文件路径

MIN_SENT_LEN = 20     # 过短句合并阈值
MAX_SENT_LEN = 500    # 过长句切分阈值
PREVIEW_LINES = 5     # 每个文件预览的切分文本行数
MIN_LINE_LEN = 10     # 过滤过短行的阈值

# =========================
# 文本清洗工具函数（新增非中文字符过滤）
# =========================
def filter_non_chinese(text: str) -> str:
    """
    过滤文本中的非中文字符，保留中文、中文标点、数字、常用符号
    """
    pattern = re.compile(r"[^\u4e00-\u9fa5\u3000-\u303f\uff00-\uffef0-9\.\,\!\?\;\:\'\"\"\(\)\（\）\【\】\-\—\、\·\《\》]")
    return pattern.sub("", text)

# =========================
# HTML 清洗 & 文本提取函数
# =========================
def clean_html_structure(html: str) -> BeautifulSoup:
    """
    清洗 HTML 结构，移除无关标签和垃圾内容
    """
    soup = BeautifulSoup(html, "lxml")

    # 移除功能性标签
    for tag in soup(["script", "style", "nav", "footer", "header", "aside", "form", "iframe"]):
        tag.decompose()

    # 移除包含垃圾关键词的 div
    garbage_keywords = ["nav", "menu", "footer", "header", "breadcrumb", "language", "search", "share", "comment"]
    for div in soup.find_all("div"):
        if not hasattr(div, "attrs") or div.attrs is None:
            continue

        classes = div.attrs.get("class") or []
        if not isinstance(classes, list):
            classes = [classes] if classes else []

        div_id = div.attrs.get("id") or ""
        attrs_text = " ".join(classes).lower() + " " + div_id.lower()

        if any(keyword in attrs_text for keyword in garbage_keywords):
            div.decompose()

    return soup

def extract_policy_main(soup: BeautifulSoup) -> str:
    """
    精准提取政策文档的正文内容
    """
    candidates = [
        {"name": "article"},
        {"name": "div", "class_": "TRS_Editor"},
        {"name": "div", "class_": "article-content"},
        {"name": "div", "id": "content"},
        {"name": "div", "class_": "content"},
    ]

    for candidate in candidates:
        node = soup.find(**candidate)
        if node:
            text = node.get_text(separator="\n", strip=True)
            if len(text) > 200:
                return text

    return soup.body.get_text(separator="\n", strip=True) if soup.body else ""

def clean_policy_text(text: str) -> list[str]:
    """
    清洗政策文本，过滤垃圾行和无效内容
    - 新增：非中文字符过滤 + 垃圾关键词黑名单
    """
    if not text:
        return []

    # 新增：针对当前垃圾内容的关键词黑名单
    garbage_keywords = [
        "12345网上接诉即办", "抱歉，没听清", "请再说一遍",
        "РУССКИЙ ЯЗЫК", "俄语", "ENGLISH", "政府公报"
    ]

    # 原有垃圾短语列表
    garbage_phrases = [
        "网站识别码", "ICP备", "公网安备", "回到顶部", "附件下载",
        "主办单位", "承办单位", "无障碍", "多语种", "登录",
        "国务院各部门网站", "版权所有", "隐私政策", "联系方式"
    ]

    clean_lines = []
    for line in text.split("\n"):
        # 步骤1：过滤非中文字符（核心修复）
        line = filter_non_chinese(line)

        # 步骤2：去除多余空格并清洗
        line = re.sub(r"\s+", " ", line).strip()

        # 步骤3：过滤过短的行
        if len(line) < MIN_LINE_LEN:
            continue

        # 步骤4：过滤垃圾关键词/短语
        if any(kw in line for kw in garbage_keywords) or any(ph in line for ph in garbage_phrases):
            continue

        # 步骤5：过滤重复关键词过多的行
        if line.count("农业农村") > 5:
            continue

        clean_lines.append(line)

    return clean_lines

# =========================
# 句子级切分函数
# =========================
def split_chinese_sentences(text: str):
    """
    按中文标点切分句子
    """
    text = re.sub(r"\s+", " ", text.strip())
    parts = re.split(r"(。|！|？|\n)", text)

    sentences = []
    buffer = ""

    for part in parts:
        if part in ["。", "！", "？", "；", "\n"]:
            buffer += part
            sentences.append(buffer.strip())
            buffer = ""
        else:
            buffer += part

    if buffer.strip():
        sentences.append(buffer.strip())

    return sentences

def normalize_sentences(sentences):
    """
    合并过短句，拆分过长句
    """
    merged = []
    buffer = ""

    for s in sentences:
        if len(s) < MIN_SENT_LEN:
            buffer += s
        else:
            if buffer:
                s = buffer + s
                buffer = ""
            merged.append(s)

    if buffer:
        merged.append(buffer)

    final = []
    for s in merged:
        if len(s) > MAX_SENT_LEN:
            for i in range(0, len(s), MAX_SENT_LEN):
                final.append(s[i:i + MAX_SENT_LEN])
        else:
            final.append(s)

    return final

# =========================
# 预览 & 导出函数
# =========================
def preview_chunks(chunks: list, fname: str):
    """
    预览切分后的文本块
    """
    print(f"\n===== {fname} 切分结果预览（前 {PREVIEW_LINES} 条） =====")
    if len(chunks) == 0:
        print("  无有效文本块")
        return

    preview_num = min(PREVIEW_LINES, len(chunks))
    for i in range(preview_num):
        print(f"  [{i+1}] {chunks[i]}")

    if len(chunks) > preview_num:
        print(f"  ... 共 {len(chunks)} 条，仅显示前 {preview_num} 条")
    print("="*50 + "\n")

def export_faiss_data(vectorstore: FAISS, export_path: str):
    """
    导出 FAISS 向量库中的所有文本和元数据
    """
    if vectorstore is None:
        print("FAISS 向量库为空，无需导出")
        return

    print(f"\n开始导出 FAISS 数据到 {export_path}...")

    try:
        all_docs = vectorstore.get_documents()
    except AttributeError:
        all_docs = list(vectorstore.docstore._dict.values())

    exported_data = []
    for doc in all_docs:
        exported_data.append({
            "page_content": doc.page_content,
            "metadata": doc.metadata,
            "length": len(doc.page_content)
        })

    with open(export_path, "w", encoding="utf-8") as f:
        json.dump(exported_data, f, ensure_ascii=False, indent=2)

    print(f"✅ 成功导出 {len(exported_data)} 条数据到 {export_path}")

# =========================
# 主逻辑
# =========================
def main():
    embeddings = OpenAIEmbeddings()

    vectorstore = None
    if os.path.exists(FAISS_INDEX_DIR):
        print("加载已有 FAISS 向量库...")
        vectorstore = FAISS.load_local(
            FAISS_INDEX_DIR,
            embeddings,
            allow_dangerous_deserialization=True
        )

    metadata = {}
    if os.path.exists(METADATA_FILE):
        with open(METADATA_FILE, "r", encoding="utf-8") as f:
            metadata = json.load(f)

    html_files = glob(os.path.join(RAG_FOLDER, "*.html"))
    if not html_files:
        raise FileNotFoundError(f"{RAG_FOLDER} 下未找到 HTML 文件")

    new_texts = []
    new_metas = []

    for file in html_files:
        fname = os.path.basename(file)
        last_modified = os.path.getmtime(file)

        if fname in metadata and metadata[fname] == last_modified:
            print(f"跳过未修改文件: {fname}")
            continue

        print(f"处理文件: {fname}")

        with open(file, "r", encoding="utf-8") as f:
            html = f.read()

        # HTML 清洗 + 正文提取 + 文本过滤
        cleaned_soup = clean_html_structure(html)
        raw_text = extract_policy_main(cleaned_soup)
        clean_lines = clean_policy_text(raw_text)
        cleaned_text = "\n".join(clean_lines)

        if not cleaned_text:
            print(f"  → 该文件无有效政策文本，跳过")
            metadata[fname] = last_modified
            continue

        # 句子切分
        sentences = split_chinese_sentences(cleaned_text)
        chunks = normalize_sentences(sentences)

        # 预览
        preview_chunks(chunks, fname)

        # 收集数据
        for chunk in chunks:
            new_texts.append(chunk)
            new_metas.append({
                "source": fname,
                "file_path": file,
                "chunk_length": len(chunk)
            })

        metadata[fname] = last_modified
        print(f"  → 清洗后切分得到 {len(chunks)} 个有效句子块")

    # 写入 FAISS
    if new_texts:
        if vectorstore is None:
            vectorstore = FAISS.from_texts(
                new_texts,
                embeddings,
                metadatas=new_metas
            )
            print(f"创建新向量库，共 {len(new_texts)} 条文本")
        else:
            vectorstore.add_texts(
                new_texts,
                metadatas=new_metas
            )
            print(f"增量添加 {len(new_texts)} 条文本")

        os.makedirs(FAISS_INDEX_DIR, exist_ok=True)
        vectorstore.save_local(FAISS_INDEX_DIR)

        with open(METADATA_FILE, "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)

        print(f"\nFAISS 向量库已保存到: {FAISS_INDEX_DIR}")
    else:
        print("\n没有新文件需要处理，向量库保持不变")

    # 导出数据
    if vectorstore is not None:
        export_faiss_data(vectorstore, EXPORT_FILE)

if __name__ == "__main__":
    main()