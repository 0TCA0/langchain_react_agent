"""
总结服务类：将用户提问和参考资料给模型进行总结回复
"""
import re
import threading

from rag.vector_store import VectorStoreService
from utils.config_handler import chroma_conf
from utils.prompt_loader import load_rag_prompts
from utils.logger_handler import logger
from langchain_core.prompts import PromptTemplate
from model.factory import get_chat_model
from langchain_core.output_parsers import StrOutputParser

class RagSummarizeService(object):
    """RAG 服务入口，负责检索、重排、总结和来源整理。"""

    def __init__(self):
        """初始化向量库、提示词链路和检索相关参数。"""
        self.vector_store = VectorStoreService()
        self._collection_ready_checked = False  # 标记是否已检查向量库就绪状态
        self._repair_lock = threading.Lock()  # 创建线程锁，防止多个请求同时触发向量库重建，保证串行执行
        self.prompt_text = load_rag_prompts()
        self.prompt_template = PromptTemplate.from_template(self.prompt_text)
        self.model = get_chat_model()
        self.chain = self._init_chain()
        self.top_k = chroma_conf["k"]  # 设置最终返回的文档数量
        self.candidate_k = chroma_conf.get("candidate_k", max(self.top_k * 2, self.top_k))  # 设置候选召回数量（用于重排）
        self.min_relevance_score = chroma_conf.get("min_relevance_score", 0.0)  # 设置最低相关性分数阈值，过滤掉相关性过低的文档，0.0（不过滤

        # 定义口语化表达与专业术语的映射，用于查询扩展，提高检索召回率
        # 如用户问："机器人不回充怎么办"，扩展为："不回充 回充失败 无法返回充电座 找不到充电座"
        self.synonym_map = {
            "不回充": ["回充失败", "无法返回充电座", "找不到充电座"],
            "回不了充": ["回充失败", "无法返回充电座"],
            "迷路": ["定位异常", "建图异常", "导航异常"],
            "漏扫": ["清扫遗漏", "覆盖率低"],
            "水痕": ["拖地水痕", "拖布湿度", "地面残留水渍"],
            "缠头发": ["毛发缠绕", "边刷缠绕", "主刷缠绕"],
            "噪音大": ["异响", "噪声异常"],
            "不出水": ["拖地不出水", "水箱异常"],
            "卡住": ["脱困失败", "避障失败"],
        }
        # 定义中文停用词集合，在提取关键词时过滤无意义词汇
        self.stopwords = {
            "的", "了", "呢", "吗", "呀", "啊", "我", "想", "请问", "一下", "怎么", "怎样",
            "是否", "一个", "这个", "那个", "可以", "需要", "有没有", "如何", "机器人", "扫地机器人",
        }

    def _init_chain(self):
        """构造“提示词 -> 模型 -> 文本解析”的最小总结链。"""
        chain = self.prompt_template | self.model | StrOutputParser()
        return chain

    def _ensure_collection_ready(self):
        """
        懒加载机制，首次使用时自动加载知识库
        向量库为空时自动触发一次本地知识入库，避免首次使用直接空检索。
        """
        if self._collection_ready_checked:
            return
        self._collection_ready_checked = True

        try:
            current_count = self.vector_store.vector_store._collection.count()  # 获取向量库中文档数量
        except Exception as e:
            logger.error(f"获取向量库文档数量失败: {str(e)}", exc_info=True)
            return

        if current_count > 0:
            logger.info(f"当前向量库已有文档，数量: {current_count}")
            return

        logger.warning("检测到向量库为空，开始自动加载知识文档")
        try:
            self.vector_store.load_document()  # 扫描 data/ 目录，切分文档，计算向量，存入 Chroma
            latest_count = self.vector_store.vector_store._collection.count()
            logger.info(f"自动加载完成，当前向量库文档数量: {latest_count}")
        except Exception as e:
            logger.error(f"自动加载知识文档失败: {str(e)}", exc_info=True)

    @staticmethod
    def _is_corrupted_index_error(error: Exception) -> bool:
        """
        判断异常是否为向量索引损坏
        返回：True 表示索引损坏，需要重建
        检测关键词：
            "hnsw segment reader"：HNSW 索引读取错误
            "nothing found on disk"：磁盘上找不到索引文件
            "error executing plan"：执行查询计划出错
        背景：Chroma 使用 HNSW 算法构建向量索引，可能因异常退出而损坏
        """
        message = str(error).lower()
        return (
            "hnsw segment reader" in message
            or "nothing found on disk" in message
            or "error executing plan" in message
        )

    def _repair_vector_store(self):
        """在检测到索引损坏时，串行重建向量库，避免并发修复。使用线程锁保证串行执行"""
        with self._repair_lock:  # 获取线程锁
            logger.warning("检测到向量索引异常，开始重建向量库")
            self.vector_store.reset_store(clear_md5=True)  # 重置向量库（删除旧数据、清空 manifest）
            self.vector_store.load_document(force_reload=True)  # 强制重新加载所有文档,force_reload=True：忽略 MD5 检查，处理所有文件
            self._collection_ready_checked = True  # 标记为已检查,避免后续请求再次触发检查
            latest_count = self.vector_store.get_collection_count()
            logger.info(f"向量库重建完成，当前文档数量: {latest_count}")

    @staticmethod
    def _normalize_query(query: str) -> str:
        """对用户问题做轻量规范化，统一一些常见别名。"""
        normalized = re.sub(r"\s+", " ", query.strip().lower())  # 将query去除首位空白、转为小写、去除多余空格
        # 定义术语替换规则，统一不同用户对同一概念的称呼
        replacements = {
            "扫拖一体": "扫拖一体机器人",
            "回充座": "充电座",
            "基站": "充电座",
            "回基站": "回充",
        }
        # 逐个应用替换规则
        for source, target in replacements.items():
            normalized = normalized.replace(source, target)
        return normalized

    def _expand_query(self, query: str) -> str:
        """把口语化问题扩展成更适合检索的表达。"""
        normalized = self._normalize_query(query)
        expansions = []
        # 遍历同义词映射表
        # 如果用户查询包含某个口语短语（如 "不回充"）
        # 将其对应的专业术语（如 ["回充失败", "无法返回充电座", ...]）加入扩展列表
        for phrase, candidates in self.synonym_map.items():
            if phrase in normalized:
                expansions.extend(candidates)
        if expansions:
            normalized = f"{normalized} {' '.join(expansions)}"  # 将扩展词拼接到原查询后面
        return normalized

    def _query_terms(self, query: str) -> set[str]:
        """提取检索关键词，供后续重排计算覆盖率。"""
        expanded = self._expand_query(query)
        terms = set()
        for term in re.findall(r"[\u4e00-\u9fff]{2,}|[a-z0-9]+", expanded):  # 匹配 2 个及以上的中文字符，匹配英文单词或数字
            if term not in self.stopwords:  # 过滤停用词
                terms.add(term)
        return terms

    @staticmethod
    def _document_terms(content: str) -> set[str]:
        """把文档内容切成词项集合，便于和 query 做简单交集比较。"""
        return set(re.findall(r"[\u4e00-\u9fff]{2,}|[a-z0-9]+", content.lower()))

    def _rerank_score(self, query_terms: set[str], content: str, relevance_score: float) -> float:
        """
        组合向量分数和关键词覆盖率。

        这里不是完整 reranker，而是一个成本很低的启发式重排，
        用来避免“向量相似但关键词没对上”的片段排得过高。

        作用：计算综合重排分数
        策略：结合向量相似度（70%）+ 关键词覆盖率（30%）
        目的：弥补纯向量检索的不足，提升相关性
        """
        doc_terms = self._document_terms(content)
        overlap = len(query_terms & doc_terms)
        coverage = overlap / max(len(query_terms), 1)
        return relevance_score * 0.7 + coverage * 0.3

    def retriever_docs(self, query):
        """
        执行检索主流程：查询扩展 -> 候选召回 -> 轻量重排 -> 截断返回。
        返回：重排后的 top-k 文档列表
        """
        self._ensure_collection_ready()
        expanded_query = self._expand_query(query)  # 用于向量检索（包含同义词）
        query_terms = self._query_terms(query)  # 用于重排计算覆盖率

        """
        作用：执行向量相似度搜索
        参数：
            expanded_query：扩展后的查询
            k=self.candidate_k：召回 10 个候选文档
        返回值：List[Tuple[Document, float]]，每个元素包含文档和相关性分数
        """
        try:
            candidates = self.vector_store.vector_store.similarity_search_with_relevance_scores(
                expanded_query,
                k=self.candidate_k,
            )
        except Exception as e:
            logger.error(f"向量检索失败: {str(e)}", exc_info=True)
            if self._is_corrupted_index_error(e):  # 判断是否为索引损坏
                try:
                    self._repair_vector_store()  # 重建向量库
                    # 重建后重新检索
                    candidates = self.vector_store.vector_store.similarity_search_with_relevance_scores(
                        expanded_query,
                        k=self.candidate_k,
                    )
                except Exception as repair_error:
                    logger.error(f"重建后检索仍失败: {str(repair_error)}", exc_info=True)
                    return []
            else:
                return []

        # 先保留候选，再按自定义分数重新排序。
        # 存储重排后的文档和分数
        scored_docs = []
        for doc, relevance_score in candidates:
            if relevance_score < self.min_relevance_score:  # 过滤低于阈值的文档
                continue
            # 计算重排分数（向量相似度 70% + 关键词覆盖率 30%）
            rerank_score = self._rerank_score(query_terms, doc.page_content, relevance_score)
            # 将分数写入元数据
            doc.metadata["relevance_score"] = round(float(relevance_score), 4)
            doc.metadata["rerank_score"] = round(float(rerank_score), 4)
            scored_docs.append((doc, rerank_score))
        # 按重排分数降序排序,lambda item: item[1]：以分数（第二个元素）为排序键
        scored_docs.sort(key=lambda item: item[1], reverse=True)
        # 取前 top_k 个文档（4 个）,列表推导式：只提取文档对象，丢弃分数
        docs = [doc for doc, _ in scored_docs[: self.top_k]]
        logger.info(
            f"RAG检索完成，原始query={query}，扩展query={expanded_query}，候选数={len(candidates)}，入选数={len(docs)}"
        )
        return docs

    @staticmethod
    def _format_references(docs) -> str:
        """
        把命中的来源整理成回答尾部可展示的引用列表。
        作用：生成参考文献列表
        用途：让用户知道回答基于哪些资料
        """
        references = []  # 存储引用文本
        seen = set()     # 存储去重集合
        """
        作用：提取文档的来源和页码
        注意：PDF 可能有页码，TXT 没有
        """
        for doc in docs:
            source = doc.metadata.get("source", "未知来源")
            page = doc.metadata.get("page")
            """
            作用：格式化引用文本
            逻辑：
                如果有页码："扫地机器人100问.pdf 第5页"
                如果没有页码："故障排除.txt"
            """
            ref = f"{source} 第{page + 1}页" if isinstance(page, int) else source
            # 去重后添加到引用列表，避免同一来源重复显示
            if ref not in seen:
                seen.add(ref)
                references.append(ref)
        if not references:
            return ""
        return "\n参考来源：\n- " + "\n- ".join(references)
    """
    示例返回：
        参考来源：
            - 扫地机器人100问.pdf 第5页
            - 故障排除.txt
            - 维护保养.txt
    """

    def rag_summarize(self, query):
        """对外暴露的 RAG 总入口，返回“总结结果 + 引用来源”。"""
        try:
            context_docs = self.retriever_docs(query)  # 调用检索方法获取相关文档
        except Exception as e:
            logger.error(f"RAG检索流程异常: {str(e)}", exc_info=True)
            return "知识库检索暂时不可用，请稍后重试。"

        if not context_docs:
            return "未检索到相关参考资料。"

        # 把命中文档拼成可追踪来源的上下文，便于模型总结时引用。
        context_parts = []  # 存储格式化的文档片段
        for counter, doc in enumerate(context_docs, start=1):  # 遍历每个文档，从 1 开始编号
            source = doc.metadata.get("source", "未知来源")
            page = doc.metadata.get("page")
            chunk_index = doc.metadata.get("chunk_index")
            # 构建位置信息列表，示例：["来源=故障排除.txt", "切片=5"]
            location_parts = [f"来源={source}"]
            if page is not None:
                location_parts.append(f"页码={page}")
            if chunk_index is not None:
                location_parts.append(f"切片={chunk_index}")
            # 构建格式化后的文档片段，示例：[参考资料1] 来源=故障排除.txt 页码=5 | 切片=5\n文档内容
            context_parts.append(
                f"[参考资料{counter}] {' | '.join(location_parts)}\n{doc.page_content.strip()}"
            )
        context = "\n\n".join(context_parts)  # 用双换行符连接所有参考资料
        # 调用模型生成回答
        try:
            answer = self.chain.invoke(
                {
                    "input": query,
                    "context": context,
                }
            )
            return answer.strip() + self._format_references(context_docs)
        except Exception as e:
            logger.error(f"RAG总结失败: {str(e)}", exc_info=True)
            return "知识总结暂时不可用，请稍后重试。"
"""
返回拼接回答和引用来源
示例：
     当扫地机器人无法回充时，请按以下步骤排查：
      1. 检查充电座电源是否正常
      2. 确认充电座指示灯状态
      3. 清理机器人底部充电触点
      
      参考来源：
      - 故障排除.txt
      - 扫地机器人100问.pdf 第12页
"""

if __name__ == '__main__':
    rag = RagSummarizeService()
    print(rag.rag_summarize("小户型适合什么扫地机器人"))
