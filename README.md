# 🤖 扫地机器人智能客服系统

基于 LangChain + RAG + ReAct Agent 架构的智能客服系统，专为扫地机器人产品设计，支持多轮对话、知识库问答、故障诊断和维护提醒等功能。

## ✨ 功能特性

- **🔍 RAG 知识库问答**：基于 ChromaDB 向量数据库，支持 PDF/TXT 文档的智能检索
- **🤖 ReAct Agent 架构**：支持多工具调用、流式输出、多轮对话上下文管理
- **📚 多源知识融合**：支持故障排除、维护保养、选购指南等多类文档
- **🎯 智能查询扩展**：内置同义词映射和关键词提取，提升检索准确率
- **📖 引用溯源**：回答附带参考来源，支持点击查看原文片段
- **💬 流式输出**：实时显示生成过程，支持逐字打字机效果
- **🔄 会话管理**：支持多会话切换、历史记录持久化、快捷问题
- **📊 自动修复机制**：向量索引损坏时自动重建，保障系统稳定性

## 🏗️ 技术架构

```
┌─────────────────────────────────────────────────────┐
│                   Streamlit UI                      │
│  (会话管理 | 快捷问题 | 流式展示 | 引用预览)         │
└─────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────┐
│                  ReactAgent                         │
│  (消息标准化 | 事实提取 | 工具调度 | 流式执行)       │
└─────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────┐
│               Tools & Middleware                    │
│  (RAG检索 | 天气查询 | 用户画像 | 报告生成)          │
└─────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────┐
│             RagSummarizeService                     │
│  (查询扩展 | 候选召回 | 轻量重排 | 来源整理)         │
└─────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────┐
│            VectorStoreService (ChromaDB)            │
│  (文档加载 | 向量计算 | 相似度搜索 | MD5校验)        │
└─────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────┐
│              DashScope 通义千问模型                  │
│  (Qwen3-Max 聊天 | text-embedding-v4 向量)          │
└─────────────────────────────────────────────────────┘
```


## 📋 环境要求

- **Python**: >= 3.10
- **操作系统**: Windows / macOS / Linux
- **API 密钥**: DashScope API Key（通义千问）

## 🚀 快速开始

### 1. 克隆项目

```bash
git clone https://github.com/your-username/langchain-agent.git
cd langchain-agent
```


### 2. 安装依赖

```bash
pip install -r requirements.txt
```


### 3. 配置环境变量

复制环境变量模板并填入您的 DashScope API Key：

```bash
# Windows
copy .env.example .env

# Linux/macOS
cp .env.example .env
```


编辑 `.env` 文件：
```env
DASHSCOPE_API_KEY=your_actual_api_key_here
```


### 4. 启动应用

```bash
streamlit run app.py
```


应用将在浏览器中自动打开，默认地址：`http://localhost:8501`

### 5. 首次使用

1. 系统会自动加载 `data/` 目录下的知识文档到向量数据库
2. 可使用快捷问题快速体验
3. 或直接输入您的问题开始对话

## 📁 项目结构

```
langchain-agent-master/
├── app.py                          # Streamlit 主应用入口
├── agent/                          # Agent 核心模块
│   ├── react_agent.py              # ReAct Agent 实现
│   └── tools/                      # 工具集
│       ├── agent_tools.py          # RAG、天气、用户画像等工具
│       └── middleware.py           # 中间件（监控、日志、提示词切换）
├── rag/                            # RAG 检索模块
│   ├── rag_service.py              # RAG 服务（检索、重排、总结）
│   └── vector_store.py             # ChromaDB 向量数据库封装
├── model/                          # 模型工厂
│   └── factory.py                  # 聊天模型和 Embedding 模型初始化
├── utils/                          # 工具类
│   ├── bootstrap.py                # 运行时环境检查
│   ├── chat_session_store.py       # 会话持久化管理
│   ├── config_handler.py           # YAML 配置加载
│   ├── file_handler.py             # 文件处理（PDF/TXT 解析）
│   ├── logger_handler.py           # 日志配置
│   ├── path_tool.py                # 路径处理
│   └── prompt_loader.py            # 提示词加载
├── config/                         # 配置文件
│   ├── agent.yaml                  # Agent 配置（外部数据路径）
│   ├── chroma.yaml                 # ChromaDB 配置（分块参数、检索参数）
│   ├── prompt.yaml                 # 提示词路径配置
│   └── rag.yaml                    # RAG 模型配置
├── prompts/                        # 提示词模板
│   ├── main_prompt.txt             # 主系统提示词
│   ├── rag_summarize.txt           # RAG 总结提示词
│   └── report_prompt.txt           # 报告生成提示词
├── data/                           # 知识库文档
│   ├── 扫地机器人100问.pdf
│   ├── 扫拖一体机器人100问.txt
│   ├── 故障排除.txt
│   ├── 维护保养.txt
│   ├── 选购指南.txt
│   └── external/
│       └── records.csv             # 外部数据
├── storage/                        # 存储目录
│   ├── chroma_db/                  # ChromaDB 向量数据库
│   ├── chat_sessions.json          # 会话历史
│   └── knowledge_manifest.json     # 知识文档清单
├── logs/                           # 日志目录
├── requirements.txt                # Python 依赖
├── .env.example                    # 环境变量模板
└── README.md                       # 项目说明文档
```


## ⚙️ 配置说明

### chroma.yaml（向量数据库配置）

| 参数                        | 说明             | 默认值            |
| --------------------------- | ---------------- | ----------------- |
| `collection_name`           | 向量集合名称     | agent             |
| `persist_directory`         | 数据库持久化路径 | storage/chroma_db |
| `k`                         | 最终返回文档数量 | 4                 |
| `candidate_k`               | 候选召回数量     | 10                |
| `min_relevance_score`       | 最低相关性阈值   | 0.0               |
| `data_path`                 | 知识文档目录     | data              |
| `chunk_size`                | 通用分块大小     | 240               |
| `txt_chunk_size`            | TXT 分块大小     | 220               |
| `pdf_chunk_size`            | PDF 分块大小     | 420               |
| `allow_knowledge_file_type` | 支持的文档类型   | ["txt", "pdf"]    |

### rag.yaml（RAG 模型配置）

| 参数                   | 说明         | 默认值            |
| ---------------------- | ------------ | ----------------- |
| `chat_model_name`      | 聊天模型名称 | qwen3-max         |
| `embedding_model_name` | 向量模型名称 | text-embedding-v4 |

## 🛠️ 核心功能详解

### RAG 检索流程

1. **查询规范化**：统一术语（如"扫拖一体" → "扫拖一体机器人"）
2. **查询扩展**：同义词映射（如"不回充" → "回充失败 无法返回充电座"）
3. **候选召回**：向量相似度搜索，召回 top-10 候选文档
4. **轻量重排**：向量分数（70%）+ 关键词覆盖率（30%）
5. **截断返回**：取 top-4 文档供模型总结
6. **引用整理**：生成带来源标记的回答

### 内置工具列表

| 工具                         | 功能                 |
| ---------------------------- | -------------------- |
| `rag_summarize`              | RAG 知识库检索与总结 |
| `get_weather`                | 查询城市天气         |
| `get_user_location`          | 获取用户所在城市     |
| `get_user_id`                | 获取用户 ID          |
| `get_user_profile`           | 获取用户画像         |
| `get_current_month`          | 获取当前月份         |
| `list_report_months`         | 列出报告可用月份     |
| `fetch_external_data`        | 获取外部数据         |
| `fetch_latest_external_data` | 获取最新外部数据     |
| `fill_context_for_report`    | 填充报告上下文       |

## 📖 使用指南

### 对话示例

```
用户: 扫地机器人不回充了怎么排查？

助手: 当扫地机器人无法回充时，请按以下步骤排查：
      1. 检查充电座电源是否正常
      2. 确认充电座指示灯状态
      3. 清理机器人底部充电触点
      4. 检查回充路径是否有障碍物
      5. 尝试重启机器人和充电座
      
      参考来源：
      - 故障排除.txt
      - 扫地机器人100问.pdf 第12页
```


### 快捷问题

系统预设了三个常用问题，点击即可快速提问：
- 我家适合买扫拖一体还是纯扫地？
- 机器人不回充了怎么排查？
- 怎么做日常维护延长寿命？

### 会话管理

- **新建会话**：侧边栏点击"新建会话"按钮
- **切换会话**：点击侧边栏历史会话列表
- **删除会话**：点击"删除当前会话"按钮
- **清空消息**：点击主页面"清空会话"按钮

### 知识库管理

- **重建知识库**：点击"重建知识库"按钮，强制重新加载所有文档
- **添加新文档**：将 PDF/TXT 文件放入 `data/` 目录后点击重建
- **查看引用**：点击回答下方的引用标签，展开查看原文片段

## 🔧 故障排查

### 常见问题

**Q: 启动时提示缺少 DASHSCOPE_API_KEY**

> A: 请确保已创建 .env 文件并填入有效的 DashScope API Key

**Q: 知识库加载失败**

> A: 检查 data/ 目录下是否有支持的文档（.txt 或 .pdf）

**Q: 回答内容不准确**

> A: 尝试点击"重建知识库"按钮，确保向量数据库已正确加载文档

**Q: 向量检索报错（HNSW index error）**

> A: 系统会自动检测并重建损坏的索引，无需手动干预

### 日志查看

所有运行日志保存在 `logs/` 目录下，按日期命名：
```
logs/
├── agent_20260415182958.log
├── agent_20260415183526.log
└── agent_20260417.log
```


## 🤝 贡献指南

欢迎提交 Issue 和 Pull Request！

## 📄 许可证

本项目仅供学习和研究使用。

## 📧 联系方式

如有问题或建议，请通过 Issue 反馈。

---

**祝您使用愉快！**
