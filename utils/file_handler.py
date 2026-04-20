"""
计算文件 md5、返回特定文件后缀的文件路径、加载文件、文本清洗、把 FAQ/问答类长文本拆成独立的“问题\n答案”文档
"""
import hashlib
import os
import re

from langchain_core.documents import Document

from utils.logger_handler import logger
from langchain_community.document_loaders import PyPDFLoader, TextLoader


def get_file_md5_hex(file_path: str) -> str | None:
    """计算文件 md5，用于知识库增量入库判断。"""
    if not os.path.exists(file_path):
        logger.error(f"文件不存在: {file_path}")
        return None
    if not os.path.isfile(file_path):
        logger.error(f"路径不是一个文件: {file_path}")
        return None
    md5_obj = hashlib.md5()
    chunk_size = 4096  # 每次读取文件的字节数
    try:
        with open(file_path, 'rb') as f:  # 以只读二进制模式打开文件
            while chunk := f.read(chunk_size):  # chunk := ...（海象运算符）。它会在判断条件前，将 f.read(chunk_size) 的结果赋值给变量 chunk
                md5_obj.update(chunk)
            md5_hex = md5_obj.hexdigest()
            return md5_hex
    except Exception as e:
        logger.error(f"计算文件{file_path}md5失败, {str(e)}")
        return None


def listdir_with_allowed_type(path, allowed_types):
    """递归扫描目录，只返回允许后缀的文件路径。"""
    files = []
    if not os.path.isdir(path):
        logger.error(f"路径不是一个目录: {path}")
        return tuple()

    for root, _, filenames in os.walk(path):
        for filename in filenames:
            if filename.endswith(allowed_types):
                files.append(os.path.join(root, filename))

    return tuple(sorted(files))  # 转成元组，不允许修改


def pdf_loader(file_path, password=None) -> list[Document]:
    """加载 PDF 文件并转成 LangChain Document 列表。"""
    return PyPDFLoader(file_path=file_path, password=password).load()


def txt_loader(file_path) -> list[Document]:
    """加载 UTF-8 文本文件并转成 LangChain Document 列表。"""
    return TextLoader(file_path, encoding="utf-8").load()


def clean_text(text: str) -> str:
    """做轻量文本清洗，统一空白、换行和 BOM。"""
    """
    功能：对原始文本进行深度清洗。包括去除 BOM 头（\ufeff）、统一换行符（\r\n 转 \n）、去除多余空白字符、压缩连续空行等。
    作用：PDF 或网页抓取的文本通常包含大量排版噪音。清洗后的文本能显著降低大模型的幻觉率，并提高 Embedding 向量的质量
    """
    if not text:
        return ""

    cleaned = text.replace("\ufeff", "").replace("\u3000", " ")  # 将Unicode BOM（Byte Order Mark）字符（其出现在文本开头）和Unicode全角空格（中文排版中常用的“宽空格”）替换为空字符串
    cleaned = cleaned.replace("\r\n", "\n").replace("\r", "\n")
    cleaned = re.sub(r"[ \t]+", " ", cleaned)     # 将多个连续空格替换成一个空格
    cleaned = re.sub(r" *\n *", "\n", cleaned)    # 清理换行符前后的空格，替换成只有一个换行符
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)  # 将 3 次或更多的换行符替换成一个空行（\n\n 表示一个段落结束，中间留一行空隙）
    return cleaned.strip()


def normalize_documents(documents: list[Document]) -> list[Document]:
    """对一组文档做统一清洗，并过滤空内容。"""
    normalized = []
    for document in documents:
        cleaned = clean_text(document.page_content)
        if not cleaned:
            continue
        document.page_content = cleaned
        normalized.append(document)
    return normalized


def split_qa_documents(documents: list[Document]) -> list[Document]:
    """
    功能：利用正则表达式识别类似“问题...？\n- 答案...”的结构，将长篇的 FAQ 文档拆分成独立的“一问一答”小文档。
    作用：这是提升检索命中率的核心手段。如果不拆分，切块时可能会把“问题”和“答案”切分到不同的块里，导致检索失败。拆分后，用户问什么问题，就能精准命中对应的答案块。

    把 FAQ/问答类长文本拆成独立的“问题-答案”文档。

    这样做的目标是提升知识库命中率，避免一整个 FAQ 文件被当成长文切碎后难以命中。
    """
    qa_documents = []
    pattern = re.compile(
        r"(?ms)(?:^|\n)(?:\d+\.\s*)?(?:\*\*)?(?P<question>[^\n？?]{3,}[？?])(?:\*\*)?\s*\n-\s*(?P<answer>.*?)(?=(?:\n(?:\d+\.\s*)?(?:\*\*)?[^\n？?]{3,}[？?](?:\*\*)?\s*\n-\s)|\Z)"
    )  # 寻找以“？”结尾的文本作为 question，以及紧随其后（通常以 - 开头）的文本作为 answer

    for document in documents:
        matches = list(pattern.finditer(document.page_content))  # 查找所有符合正则表达式的片段，将其转换为列表
        if len(matches) < 3:
            qa_documents.append(document)
            continue

        for index, match in enumerate(matches):
            question = clean_text(match.group("question"))
            answer = clean_text(match.group("answer"))
            if not question or not answer:
                continue
            qa_documents.append(
                Document(
                    page_content=f"问题：{question}\n答案：{answer}",
                    metadata={**document.metadata, "qa_index": index},
                )
            )

    return qa_documents
