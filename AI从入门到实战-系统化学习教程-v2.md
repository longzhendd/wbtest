# AI 从入门到实战：系统化学习教程

> **版本**: v2.0 | **更新日期**: 2025-06 | **适用人群**: 零基础~中级开发者、产品经理、技术决策者
>
> **v2.0 更新日志**: 修正版本日期 | 新增第0周预算规划 | 升级RAG代码为LangChain最新API | 新增AI安全与红队测试模块 | 补充毕业项目评估矩阵 | 补充Multi-Agent评估环节 | 新增第六部分深度专题（Agent架构选型/RAG升级路线/12周定制化策略）

---

## 目录

- [第一部分：行业教程全景对比](#第一部分行业教程全景对比)
- [第二部分：AI 核心知识体系](#第二部分ai-核心知识体系)
- [第三部分：12 周系统学习路线](#第三部分12-周系统学习路线)
- [第四部分：实战项目手册](#第四部分实战项目手册)
- [第五部分：工具与资源速查](#第五部分工具与资源速查)
- [第六部分：深度专题](#第六部分深度专题)
- [附录](#附录)

---

# 第一部分：行业教程全景对比

## 1.1 全球 Top 10 AI 课程横向评测

| # | 课程名称 | 平台 | 难度 | 时长 | 价格 | 最佳受众 | 核心特色 |
|---|---------|------|------|------|------|---------|---------|
| 1 | AI For Everyone | DeepLearning.AI / Coursera | ⭐ 入门 | 6h | 免费 | 非技术人员、管理者 | 无代码门槛，建立 AI 战略视野 |
| 2 | Machine Learning (Andrew Ng) | Coursera | ⭐⭐ 初中 | 60h | 免费旁听 | 开发者、数据科学家 | 经典入门课，700万+学员 |
| 3 | CS50's Intro to AI with Python | HarvardX / edX | ⭐⭐ 中级 | 70h | 免费 | 有编程基础者 | 项目驱动，每模块一个实战 |
| 4 | AI Nanodegree | Udacity | ⭐⭐ 中级 | 120h | $249/月 | 转行者、求职者 | 导师+简历+项目集 |
| 5 | Deep Learning Specialization | DeepLearning.AI / Coursera | ⭐⭐⭐ 中高 | 150h | 免费旁听 | 深度学习方向专精 | 5门课系列，覆盖 CNN/RNN/Seq |
| 6 | Generative AI for Everyone | DeepLearning.AI / Coursera | ⭐ 入门 | 8h | 免费 | 关注生成式AI者 | 涵盖 GPT/DALL·E/RAG/安全 |
| 7 | LangChain for LLM Apps | LangChain University | ⭐⭐ 中级 | 20h | 免费 | LLM 应用开发者 | 框架实战：链/Agent/记忆/向量库 |
| 8 | AI Engineering Professional Certificate | DataCamp | ⭐⭐⭐ 中高 | 80h | 订阅制 | MLOps 工程师 | 生产部署：Docker/FastAPI/监控 |
| 9 | AI Graduate Certificate | Stanford Online | ⭐⭐⭐⭐ 高级 | 200h+ | $15,000+ | 研究者、技术负责人 | 斯坦福课程同等难度 |
| 10 | Practical Deep Learning for Coders | fast.ai | ⭐⭐ 中级 | 40h | 免费 | 动手型开发者 | "先跑通再理解"，top-down 教学 |

## 1.2 课程模式分类

| 模式 | 代表课程 | 优势 | 劣势 | 适合谁 |
|------|---------|------|------|--------|
| **自上而下 (Top-Down)** | fast.ai | 即时成就感，先跑通再理解 | 基础可能不牢 | 急于出成果的开发者 |
| **自下而上 (Bottom-Up)** | Andrew Ng ML | 基础扎实，数学清晰 | 周期长，前期枯燥 | 追求深度理解的学习者 |
| **项目驱动 (Project-Based)** | CS50 AI, Udacity | 实战感强，作品集积累 | 理论可能跳过 | 转行者、求职者 |
| **学术严谨 (Academic)** | Stanford, CS229 | 深度+前沿+人脉 | 门槛高、耗时长 | 研究者、PhD |
| **工程导向 (Engineering)** | DataCamp, LangChain | 生产级技能，直接可用 | 视野偏窄 | 在职工程师 |

## 1.3 中文生态 Top 5 教程

| # | 名称 | 平台 | 特色 | 受众 |
|---|------|------|------|------|
| 1 | 李宏毅机器学习 | YouTube / B站 | 中文最全ML课，幽默易懂 | 中文学习者首选 |
| 2 | 跟李沐学AI | B站 / 动手学深度学习 | 代码+论文逐行讲解 | 动手型学习者 |
| 3 | AI Agent & RAG 12周路线 | CSDN | 8模块37论文，3阶段递进 | Agent 方向工程师 |
| 4 | 提示工程完整指南 | LearnPrompting (中文版) | Prompt Engineering 系统教程 | Prompt 工程师 |
| 5 | Azure AI Fundamentals | 微软学习 | 免费+认证，中文支持 | 企业级入门 |

---

# 第二部分：AI 核心知识体系

## 2.1 知识地图总览

```
AI 知识体系
│
├── 🧱 基础层
│   ├── Python 编程
│   ├── 线性代数（矩阵运算、特征值）
│   ├── 概率统计（贝叶斯、分布）
│   └── 微积分（梯度、链式法则）
│
├── 🤖 机器学习层
│   ├── 监督学习（回归、分类、SVM）
│   ├── 无监督学习（聚类、降维）
│   ├── 模型评估（交叉验证、偏差-方差）
│   └── 特征工程
│
├── 🧠 深度学习层
│   ├── 神经网络基础（前向/反向传播）
│   ├── CNN（卷积神经网络）→ 计算机视觉
│   ├── RNN/LSTM/Transformer → 序列建模
│   └── 训练技巧（正则化、优化器、调参）
│
├── 🌐 大语言模型层
│   ├── Transformer 架构深入
│   ├── 预训练 / 微调 / RLHF / DPO
│   ├── Prompt Engineering
│   └── 模型评估（Chatbot Arena 等）
│
├── ⚡ 应用工程层
│   ├── RAG（检索增强生成）
│   ├── AI Agent（ReAct / Plan-then-Execute）
│   ├── Multi-Agent 系统
│   ├── AI 安全（红队测试/对齐/护栏）
│   └── MLOps（部署/监控/评估）
│
└── 💼 产品商业层
    ├── AI Native 产品设计
    ├── AI 定价策略
    ├── 行业应用
    └── 伦理与安全
```

## 2.2 核心概念速查表

### LLM 核心概念

| 概念 | 一句话解释 | 重要性 |
|------|----------|--------|
| **Transformer** | 基于自注意力机制的序列模型架构，所有现代LLM的基础 | ⭐⭐⭐ |
| **Token** | 文本的最小处理单元，约4个字符≈1个Token | ⭐⭐⭐ |
| **预训练** | 在海量文本上学习语言模式（自监督） | ⭐⭐⭐ |
| **SFT (监督微调)** | 用标注数据教模型遵循指令 | ⭐⭐ |
| **RLHF** | 用人类反馈的强化学习，对齐模型输出偏好 | ⭐⭐⭐ |
| **DPO** | 直接偏好优化，RLHF的简化替代 | ⭐⭐ |
| **Temperature** | 控制输出随机性（0=确定，1=随机） | ⭐⭐ |
| **Top-p** | 核采样，控制候选Token范围 | ⭐⭐ |
| **幻觉** | 模型生成看似合理但实际错误的内容 | ⭐⭐⭐ |

### Agent 核心概念

| 概念 | 一句话解释 | 重要性 |
|------|----------|--------|
| **ReAct** | "边想边做"——Thought-Action-Observation循环 | ⭐⭐⭐ |
| **Plan-then-Execute** | "先想再做"——Planner/Executor分离，更可控 | ⭐⭐⭐ |
| **Function Calling** | LLM调用外部工具/函数的能力 | ⭐⭐⭐ |
| **MCP** | Model Context Protocol，模型与工具通信标准 | ⭐⭐ |
| **A2A** | Agent-to-Agent，智能体间通信协议 | ⭐⭐ |
| **Multi-Agent** | 多个Agent协作完成复杂任务 | ⭐⭐ |
| **Workflow Agent** | 人工编排流程的Agent（可控） | ⭐⭐⭐ |
| **Autonomous Agent** | 自主决策的Agent（灵活） | ⭐⭐ |

### RAG 核心概念

| 概念 | 一句话解释 | 重要性 |
|------|----------|--------|
| **向量检索** | 将文本转为向量，通过相似度搜索 | ⭐⭐⭐ |
| **Chunking** | 将文档切分为可检索的片段 | ⭐⭐⭐ |
| **HyDE** | 假设性文档嵌入，先生成假设答案再检索 | ⭐⭐ |
| **GraphRAG** | 用知识图谱增强检索，擅长全局性问题 | ⭐⭐⭐ |
| **Agentic RAG** | Agent驱动的检索，多步迭代 | ⭐⭐ |
| **RAGAS** | RAG评估框架（Faithfulness/Relevancy等） | ⭐⭐ |

### AI 安全核心概念

| 概念 | 一句话解释 | 重要性 |
|------|----------|--------|
| **Prompt 注入** | 通过恶意输入覆盖系统指令，劫持模型行为 | ⭐⭐⭐ |
| **越狱 (Jailbreak)** | 绕过模型安全限制，获取被禁止的输出 | ⭐⭐⭐ |
| **数据泄露** | 模型在输出中泄露训练数据中的敏感信息 | ⭐⭐⭐ |
| **红队测试** | 模拟攻击者对AI系统进行安全测试 | ⭐⭐⭐ |
| **护栏 (Guardrails)** | 输入/输出过滤层，阻止不安全内容 | ⭐⭐ |
| **对齐 (Alignment)** | 使模型行为符合人类意图和价值观 | ⭐⭐ |
| **可解释性** | 理解模型为何做出特定决策的能力 | ⭐⭐ |

---

# 第三部分：12 周系统学习路线

> **学习节奏**: 每周10小时 | **前置要求**: Python 基础 | **学习策略**: 理论30% + 实践50% + 输出20%

## ⚪ 第0周：预算规划与环境准备

> **为什么需要第0周**: 很多学习者在第3-4周因API费用、环境问题而中断。提前规划预算和环境，可显著降低流失率。

### 预算规划

| 费用类别 | 基础方案 | 标准方案 | 高级方案 |
|----------|---------|---------|---------|
| **LLM API** | OpenAI $5额度 | OpenAI $20 + DeepSeek免费 | OpenAI $50 + Anthropic $20 |
| **向量数据库** | Chroma (本地免费) | Chroma + FAISS (本地) | Pinecone (免费层) |
| **云服务** | 本地运行 | Colab Pro $10/月 | AWS/GCP $30-50/月 |
| **课程** | 全部免费旁听 | 1-2门付费认证 | 多门付费 + 纳米学位 |
| **总预算** | **$5-20** | **$50-80** | **$150-250** |

> 💡 **省钱技巧**: DeepSeek API 价格约为 OpenAI 的 1/10，适合大量练习；HuggingFace 免费模型可替代大多数 API 调用。

### 环境准备清单

| 检查项 | 工具 | 验证方式 |
|--------|------|---------|
| Python 3.11+ | python --version | 输出 3.11.x |
| pip 最新版 | pip --version | 无报错 |
| Jupyter Notebook | jupyter notebook | 浏览器打开 |
| Git | git --version | 输出版本号 |
| VS Code + 插件 | Python, Jupyter, GitLens | 插件已安装 |
| OpenAI API Key | echo $OPENAI_API_KEY | 非空 |
| HuggingFace 账号 | huggingface-cli whoami | 输出用户名 |
| Docker (可选) | docker --version | 输出版本号 |

---

## 🟢 阶段一：基础构建（第1-4周）

### 第1周：Python for AI + 开发环境搭建

| 任务 | 内容 | 产出 |
|------|------|------|
| 环境搭建 | Python 3.11+, Jupyter, VS Code, Git | 可运行的开发环境 |
| Python 核心库 | NumPy（矩阵运算）、Pandas（数据处理）、Matplotlib（可视化） | 完成数据清洗小项目 |
| AI 工具链 | OpenAI API 申请、HuggingFace 注册 | API Key 就绪 |

**实战任务**: 用 Pandas 分析一个真实数据集（如 Kaggle 的 Titanic），输出数据报告。

### 第2周：机器学习基础

| 任务 | 内容 | 产出 |
|------|------|------|
| 数学基础 | 线性回归 → 梯度下降 → 损失函数 | 手写线性回归（不用sklearn） |
| 核心算法 | 逻辑回归、决策树、随机森林、SVM | 完成 Andrew Ng ML 第1-3周 |
| 模型评估 | 训练/测试集、交叉验证、偏差-方差权衡 | 模型评估速查笔记 |

**推荐资源**:
- 🎬 Andrew Ng《Machine Learning》— Coursera 免费旁听
- 📖 《动手学深度学习》第1-3章

### 第3周：深度学习 + Transformer

| 任务 | 内容 | 产出 |
|------|------|------|
| 神经网络 | 前向传播、反向传播、激活函数 | 用NumPy手写一个2层网络 |
| CNN基础 | 卷积、池化、ResNet | 图像分类项目（CIFAR-10） |
| Transformer | 自注意力、多头注意力、位置编码 | 阅读并理解《Attention Is All You Need》 |

**推荐资源**:
- 🎬 fast.ai《Practical Deep Learning for Coders》
- 📖 Jay Alammar《The Illustrated Transformer》

### 第4周：大语言模型 (LLM) 深入

| 任务 | 内容 | 产出 |
|------|------|------|
| LLM架构 | GPT系列、BERT、LLaMA 的架构差异 | 模型架构对比笔记 |
| 训练流程 | 预训练→SFT→RLHF/DPO 的完整链路 | 训练流程图 |
| Prompt Engineering | 零样本/少样本/CoT/自我一致性 | Prompt模板库 |
| 幻觉问题 | 幻觉成因、缓解策略 | 幻觉应对手册 |

**推荐资源**:
- 🎬 OpenAI + DeepLearning.AI《ChatGPT Prompt Engineering for Developers》
- 📖 Anthropic《Prompt Engineering Guide》
- 📄 论文: A Comprehensive Survey of Hallucination in LLMs (2025)

---

## 🟡 阶段二：工程实战（第5-8周）

### 第5周：RAG 基础与进阶

| 任务 | 内容 | 产出 |
|------|------|------|
| Naive RAG | 文档加载→分块→向量化→检索→生成 | 基础RAG Demo |
| 向量数据库 | Chroma / FAISS / Pinecone 对比与选型 | 向量库选型报告 |
| Chunking策略 | 固定大小 / 语义切分 / 递归切分 | 切分策略对比实验 |
| Advanced RAG | HyDE / Reranker / 混合检索 | RAG准确率提升30%+ |

**实战项目**: 构建一个"个人知识库问答系统"——上传PDF文档，AI回答相关问题。

### 第6周：GraphRAG 与前沿 RAG

| 任务 | 内容 | 产出 |
|------|------|------|
| GraphRAG | 实体抽取→关系构建→社区检测→全局摘要 | GraphRAG Demo |
| HiRAG | 分层知识索引 | 分层检索笔记 |
| RAG评估 | RAGAS框架（Faithfulness/Relevancy/Context） | 评估报告 |
| RAG演化 | Naive→Advanced→Graph→Agentic RAG 路径图 | 演化路线图 |

**推荐论文**:
- ⭐⭐⭐ GraphRAG: From Local to Global (Microsoft, 2024)
- ⭐⭐ GeAR: Graph-enhanced Agent for RAG (2025)

### 第7周：AI Agent 开发

| 任务 | 内容 | 产出 |
|------|------|------|
| Agent原理 | ReAct / Plan-then-Execute 范式对比 | 范式对比笔记 |
| 框架实战 | LangChain + LangGraph 入门 | Agent Demo |
| Function Calling | 工具定义、参数解析、错误处理 | 天气查询Agent |
| 记忆系统 | 短期记忆 / 长期记忆 / 对话摘要 | 带记忆的对话Agent |

**实战项目**: 构建一个"智能研究助手"——给定主题，自动搜索、整理、生成报告。

**推荐资源**:
- ⭐⭐⭐ Anthropic《Building Effective Agents》(2024.12)
- 🎬 DeepLearning.AI《LangChain for LLM Application Development》

### 第8周：Multi-Agent 与评估 + AI 安全

| 任务 | 内容 | 产出 |
|------|------|------|
| Multi-Agent | CrewAI / AutoGen 角色协作 | 多Agent协作Demo |
| Agent评估 | Agent-as-a-Judge / Beyond Accuracy | 评估框架笔记 |
| 工程化 | LangSmith 追踪 / Braintrust 评估 | 可观测性Demo |
| 架构选择 | Workflow Agent vs Autonomous Agent | 架构决策指南 |
| **AI 安全基础** | **Prompt注入防御 / 护栏设计 / 红队测试入门** | **安全检查清单** |

> 🛡️ **AI 安全模块详解**:
>
> **Prompt 注入防御策略**:
> - 输入验证：限制用户输入长度和格式
> - 指令隔离：系统指令与用户输入分开传递
> - 检测层：在输出前用分类器检测恶意输出
> - 最小权限：Agent工具仅授予必要权限
>
> **护栏设计三层模型**:
> ```
> 输入护栏 → 模型护栏 → 输出护栏
> (过滤恶意提示)  (系统指令约束)  (过滤敏感输出)
> ```
>
> **红队测试入门**:
> - 目标：发现系统在恶意输入下的行为
> - 方法：角色扮演攻击、指令覆盖、编码绕过、多轮诱导
> - 工具：PromptFuzz, Garak, AIRTBench
> - 输出：漏洞报告 + 修复建议

**框架选型速查**:

| 框架 | 定位 | 适合场景 |
|------|------|---------|
| LangChain | 通用LLM应用 | 入门首选 |
| LangGraph | 状态机Agent | 复杂状态流转 |
| LlamaIndex | 数据索引/RAG | 知识密集型应用 |
| CrewAI | 多Agent协作 | 角色分工任务 |
| AutoGen | 多Agent对话 | 研究/实验 |

---

## 🔴 阶段三：产品与商业（第9-12周）

### 第9周：AI Native 产品思维

| 任务 | 内容 | 产出 |
|------|------|------|
| AI Native定义 | AI Native ≠ AI Wrapper，从0设计而非套壳 | 产品设计原则 |
| 交互范式 | 对话式 / 嵌入式 / 自主式 | 交互范式对比 |
| 案例 | Cursor / Perplexity / Claude 的AI Native设计 | 案例分析报告 |

**推荐书籍**:
- 📖 Ethan Mollick《Co-Intelligence: Living and Working with AI》
- 📖 Mustafa Suleyman《The Coming Wave》

### 第10周：AI 商业化与定价

| 任务 | 内容 | 产出 |
|------|------|------|
| 定价模式 | Token计费 / 阶梯定价 / 按结果计费 | 定价模型设计 |
| 成本结构 | API成本 / GPU成本 / 运营成本 | 单位经济学分析 |
| 商业模式 | SaaS / API / 嵌入式 / 平台 | 商业模式画布 |

### 第11周：行业应用与落地

| 行业 | 典型场景 | 代表产品 |
|------|---------|---------|
| IT服务管理 | 智能工单处理、故障自愈 | ServiceNow AI, Zendesk AI |
| 客户服务 | 多轮对话、情感分析、知识库 | Intercom Fin, Drift |
| 内容创作 | 文案、设计、代码生成 | Jasper, Copy.ai, Cursor |
| 数据分析 | 自然语言查询、自动报告 | ThoughtSpot, Equals |
| 医疗健康 | 辅助诊断、药物研发 | PathAI, Recursion |
| 教育培训 | 个性化学习、自动出题 | Khan Academy Khanmigo |

### 第12周：毕业项目与职业规划

| 任务 | 内容 | 产出 |
|------|------|------|
| 毕业项目 | 端到端构建一个AI应用 | 完整项目（含代码+文档+Demo） |
| 作品集 | 整理所有周产出 | GitHub Portfolio |
| 职业规划 | AI工程师 / AI产品经理 / AI研究员 | 职业路线图 |
| 持续学习 | 关注前沿论文、社区、Newsletter | 信息源清单 |

#### 毕业项目评估矩阵

| 维度 | 权重 | 评分标准 (1-5) | 说明 |
|------|------|---------------|------|
| **功能完整度** | 25% | 1=核心功能未完成, 5=全部功能+边界处理 | 端到端可用 |
| **代码质量** | 20% | 1=不可运行, 5=有测试+文档+类型注解 | 可维护性 |
| **架构设计** | 20% | 1=单文件脚本, 5=模块化+配置分离+错误处理 | 可扩展性 |
| **文档与演示** | 20% | 1=无文档, 5=README+API文档+演示视频 | 可理解性 |
| **创新与亮点** | 15% | 1=复刻教程, 5=有独创优化或新场景 | 差异化 |

> 🎯 **通过标准**: 加权得分 ≥ 3.5/5.0；任一维度不得低于 2 分。
>
> **自我评估方法**: 完成项目后，使用以下检查清单：
> - [ ] 别人能否在不联系你的情况下运行项目？
> - [ ] 是否覆盖了至少3种异常场景？
> - [ ] 代码是否有完整的类型注解？
> - [ ] 是否有独立的配置文件（非硬编码）？
> - [ ] README 是否包含：目的、安装、使用、架构图？

---

# 第四部分：实战项目手册

## 项目一：Prompt Engineering 实验室

**难度**: ⭐ | **时长**: 2-3小时 | **前置**: OpenAI API Key

### 目标
通过系统实验，掌握核心 Prompt 技术。

### 实验设计

```python
import openai

client = openai.OpenAI()

# 实验1: 零样本 vs 少样本
prompts = {
    "zero_shot": "将以下文本分类为正面/负面：'这家餐厅的服务太棒了！'",
    "few_shot": """将文本分类为正面/负面：
示例1: '食物难吃' → 负面
示例2: '推荐给大家' → 正面
现在分类: '这家餐厅的服务太棒了！'""",
    "cot": """将文本分类为正面/负面，先分析再回答：
'这家餐厅的服务太棒了！'"""
}

for name, prompt in prompts.items():
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    print(f"[{name}] {response.choices[0].message.content}")
```

### 实验矩阵

| 技术 | 适用场景 | 关键参数 |
|------|---------|---------|
| Zero-shot | 简单分类 | temperature=0 |
| Few-shot | 格式要求严格 | 示例3-5个 |
| Chain-of-Thought | 推理任务 | "让我们一步步思考" |
| Self-Consistency | 降低随机性 | 采样5次取多数 |
| Role-playing | 专业领域 | "你是一位资深XX专家" |

---

## 项目二：RAG 知识库问答系统

**难度**: ⭐⭐ | **时长**: 1-2天 | **前置**: Python, OpenAI API

### 架构图

```
用户提问
  ↓
Query处理（改写/扩展）
  ↓
向量检索（Chroma/FAISS）← 文档库（分块+向量化）
  ↓
Reranker 重排序
  ↓
Prompt组装（上下文 + 问题）
  ↓
LLM生成回答
  ↓
回答 + 来源引用
```

### 核心代码（LangChain 最新 API）

> ⚠️ **v2.0 更新**: 已从已废弃的 `RetrievalQA` 迁移至 `create_retrieval_chain`，这是 LangChain 0.2+ 推荐的写法。

```python
# pip install langchain langchain-openai langchain-community chromadb sentence-transformers

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# 1. 加载文档
loader = PyPDFLoader("your_document.pdf")
docs = loader.load()

# 2. 分块
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    separators=["\n\n", "\n", "。", "，", " "]
)
chunks = splitter.split_documents(docs)

# 3. 向量化
embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-zh-v1.5")
vectorstore = Chroma.from_documents(chunks, embeddings, persist_directory="./chroma_db")

# 4. 构建问答链（最新 API）
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

system_prompt = """你是一个知识库问答助手。请基于以下检索到的上下文来回答用户问题。
如果上下文中没有相关信息，请说"根据已有资料无法回答"。
不要编造信息。每个回答请引用来源片段。

上下文：
{context}"""

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}"),
])

# 4a. 文档合并链
question_answer_chain = create_stuff_documents_chain(llm, prompt)
# 4b. 检索链
rag_chain = create_retrieval_chain(vectorstore.as_retriever(search_kwargs={"k": 4}), question_answer_chain)

# 5. 问答
result = rag_chain.invoke({"input": "这个文档的主要观点是什么？"})
print(result["answer"])

# 查看来源文档
for doc in result["context"]:
    print(f"[来源: {doc.metadata}] {doc.page_content[:100]}...")
```

### API 迁移对照表

| 废弃 API (v0.1) | 推荐 API (v0.2+) | 变更说明 |
|-----------------|-----------------|---------|
| `RetrievalQA.from_chain_type()` | `create_retrieval_chain()` | 支持流式输出和更灵活的 prompt 定制 |
| `chain.invoke({"query": ...})` | `chain.invoke({"input": ...})` | 输入字段名统一为 `input` |
| `result["result"]` | `result["answer"]` | 输出字段名统一为 `answer` |
| `result["source_documents"]` | `result["context"]` | 来源文档字段名统一为 `context` |
| `load_qa_chain()` | `create_stuff_documents_chain()` | 更明确的命名 |

### 优化清单

| 维度 | 基础版 | 进阶版 | 提升 |
|------|--------|--------|------|
| 分块 | 固定500字符 | 语义切分 + 重叠 | +15% 准确率 |
| 嵌入 | 通用模型 | BGE 中文专用 | +20% 检索精度 |
| 检索 | 单路向量检索 | 混合检索(向量+BM25) | +25% 召回率 |
| 重排 | 无 | BGE Reranker | +30% 首位命中率 |
| 评估 | 人工判断 | RAGAS自动评估 | 可量化 |
| **安全** | **无** | **输入护栏+输出过滤** | **防注入/防泄露** |

---

## 项目三：AI Agent 智能助手

**难度**: ⭐⭐⭐ | **时长**: 3-5天 | **前置**: 项目二完成

### 架构设计

```
用户输入
  ↓
输入护栏 (安全过滤)
  ↓
Agent Controller (LangGraph)
  ├── 🧠 Reasoning: 分析意图，制定计划
  ├── 🔍 Search Tool: 网络搜索
  ├── 📄 RAG Tool: 知识库检索
  ├── 💻 Code Tool: 代码执行 (沙箱)
  └── 📧 Notify Tool: 发送通知
  ↓
输出护栏 (敏感信息过滤)
  ↓
输出结果 + 执行日志
```

### 核心代码 (LangGraph)

```python
# pip install langgraph langchain-openai

from typing import Annotated, TypedDict
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool

# 1. 定义工具
@tool
def search_web(query: str) -> str:
    """搜索网络获取最新信息"""
    # 实际实现可接入 Tavily/SerpAPI
    return f"搜索结果: {query} 的相关信息..."

@tool
def query_knowledge_base(question: str) -> str:
    """从知识库中检索答案"""
    # 对接项目二的 RAG 系统
    return f"知识库回答: 关于 {question} 的内容..."

tools = [search_web, query_knowledge_base]

# 2. 定义状态
class AgentState(TypedDict):
    messages: Annotated[list, lambda x, y: x + y]

# 3. 构建图
llm = ChatOpenAI(model="gpt-4o-mini").bind_tools(tools)

def agent_node(state: AgentState):
    response = llm.invoke(state["messages"])
    return {"messages": [response]}

def should_continue(state: AgentState):
    last = state["messages"][-1]
    if hasattr(last, "tool_calls") and last.tool_calls:
        return "tools"
    return END

graph = StateGraph(AgentState)
graph.add_node("agent", agent_node)
graph.add_node("tools", ToolNode(tools))
graph.add_conditional_edges("agent", should_continue)
graph.add_edge("tools", "agent")
graph.set_entry_point("agent")

app = graph.compile()

# 4. 运行
result = app.invoke({
    "messages": [{"role": "user", "content": "帮我查一下2025年AI领域的最新进展，并整理成报告"}]
})
```

### Agent 安全检查清单

| 检查项 | 说明 | 状态 |
|--------|------|------|
| 工具权限最小化 | 每个工具仅授予必要权限 | ☐ |
| 输入长度限制 | 限制用户输入最大长度 | ☐ |
| 输出敏感词过滤 | 过滤 PII/密钥/内部信息 | ☐ |
| 工具调用审计 | 记录所有工具调用日志 | ☐ |
| 执行超时 | 单次工具调用 ≤ 30秒 | ☐ |
| 人类确认 | 高风险操作需人工确认 | ☐ |

---

## 项目四：Multi-Agent 协作系统

**难度**: ⭐⭐⭐⭐ | **时长**: 5-7天 | **前置**: 项目三完成

### 场景：自动化研究团队

```
用户: "分析XX公司的投资价值"
         ↓
    ┌─────────────────────────────────┐
    │       Orchestrator (指挥者)       │
    └────────┬──────┬──────┬──────────┘
             ↓      ↓      ↓
    ┌────────┐ ┌────────┐ ┌────────┐
    │Researcher│ │Analyst│ │Writer  │
    │ (研究员) │ │(分析师)│ │(撰写者)│
    └────────┘ └────────┘ └────────┘
         ↓          ↓          ↓
    搜索资料    数据分析     报告撰写
         ↓          ↓          ↓
    ┌─────────────────────────────────┐
    │       Reviewer (审阅者)          │
    └─────────────────────────────────┘
         ↓
    最终报告
```

### CrewAI 实现

```python
# pip install crewai crewai-tools

from crewai import Agent, Task, Crew, Process
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o-mini")

# 定义Agent
researcher = Agent(
    role="资深研究员",
    goal="搜集全面、准确的信息",
    backstory="你是一位在金融领域有15年经验的研究员",
    llm=llm,
    verbose=True
)

analyst = Agent(
    role="数据分析师",
    goal="从数据中提取洞察",
    backstory="你擅长财务分析和量化建模",
    llm=llm,
    verbose=True
)

writer = Agent(
    role="技术写作专家",
    goal="将分析结果转化为专业报告",
    backstory="你擅长将复杂信息转化为清晰有力的叙述",
    llm=llm,
    verbose=True
)

# 定义Task
research_task = Task(
    description="研究{company}的业务模式、市场地位和竞争格局",
    agent=researcher,
    expected_output="结构化的研究备忘录"
)

analysis_task = Task(
    description="基于研究数据，进行财务分析和估值",
    agent=analyst,
    expected_output="财务分析报告，含关键指标和估值模型"
)

writing_task = Task(
    description="将研究和分析结果整合为投资分析报告",
    agent=writer,
    expected_output="专业的投资分析报告，含执行摘要和风险提示"
)

# 组建团队
crew = Crew(
    agents=[researcher, analyst, writer],
    tasks=[research_task, analysis_task, writing_task],
    process=Process.sequential
)

# 执行
result = crew.kickoff(inputs={"company": "某科技公司"})
print(result)
```

### Multi-Agent 输出评估框架

> 🆕 **v2.0 新增**: Multi-Agent 系统的输出质量不能仅靠"看起来对不对"来判断，需要结构化评估。

| 评估维度 | 指标 | 计算方式 | 达标线 |
|----------|------|---------|--------|
| **任务完成度** | Task Completion Rate | 完成的子任务数 / 总子任务数 | ≥ 90% |
| **信息一致性** | Cross-Agent Consistency | 各Agent产出间的矛盾数 | 0 矛盾 |
| **角色遵从度** | Role Adherence | 输出符合角色定义的比例 | ≥ 80% |
| **效率** | Total Tokens / Time | 总Token消耗 & 执行时长 | 在预算内 |
| **输出质量** | Final Output Score | 人类评审 / LLM-as-Judge 评分 | ≥ 4/5 |
| **错误恢复** | Error Recovery Rate | 出错后自动恢复的比例 | ≥ 70% |

**评估代码示例**:

```python
import json

def evaluate_crew_output(crew_result, expected_sections):
    """简易评估函数"""
    scores = {}

    # 1. 完成度：检查预期章节是否都出现
    present = sum(1 for s in expected_sections if s in str(crew_result))
    scores["completion"] = present / len(expected_sections)

    # 2. 一致性：检查是否存在自相矛盾的数值
    # (简化版：实际应用中可用LLM做语义一致性检查)
    scores["consistency"] = 1.0  # 默认通过，需人工复核

    # 3. 长度合理性：输出不应过短或过长
    output_len = len(str(crew_result))
    scores["length_score"] = 1.0 if 500 < output_len < 10000 else 0.5

    # 加权总分
    total = (scores["completion"] * 0.4 +
             scores["consistency"] * 0.3 +
             scores["length_score"] * 0.3)

    return {"scores": scores, "total": total, "pass": total >= 0.7}

# 使用
expected = ["业务模式", "市场地位", "竞争格局", "财务分析", "估值", "风险提示"]
eval_result = evaluate_crew_output(result, expected)
print(json.dumps(eval_result, indent=2, ensure_ascii=False))
```

---

# 第五部分：工具与资源速查

## 5.1 开发工具链

| 类别 | 工具 | 用途 | 入门难度 |
|------|------|------|---------|
| **LLM API** | OpenAI / Anthropic / DeepSeek | 模型调用 | ⭐ |
| **向量数据库** | Chroma / FAISS / Pinecone | RAG存储 | ⭐⭐ |
| **Agent框架** | LangChain / LangGraph / CrewAI | 应用开发 | ⭐⭐ |
| **RAG框架** | LlamaIndex / RAGAS | 知识检索 | ⭐⭐ |
| **可观测性** | LangSmith / Braintrust | 追踪评估 | ⭐⭐ |
| **部署** | Docker / FastAPI / Streamlit | 生产化 | ⭐⭐⭐ |
| **低代码** | Coze / Dify / n8n | 快速原型 | ⭐ |
| **AI安全** | Garak / PromptFuzz / NeMo Guardrails | 红队测试/护栏 | ⭐⭐⭐ |

## 5.2 必读论文清单

### ⭐⭐⭐ 必读（先看这4篇）

| 论文 | 年份 | 核心价值 | 链接 |
|------|------|---------|------|
| Attention Is All You Need | 2017 | Transformer架构，一切的基础 | [arxiv.org/abs/1706.03762](https://arxiv.org/abs/1706.03762) |
| ReAct: Synergizing Reasoning and Acting | 2022 | Agent推理范式，Thought-Action-Observation | [arxiv.org/abs/2210.03629](https://arxiv.org/abs/2210.03629) |
| GraphRAG: From Local to Global | 2024 | 知识图谱增强RAG | [arxiv.org/abs/2404.16130](https://arxiv.org/abs/2404.16130) |
| Building Effective Agents | 2024 | Agent架构设计指南（非论文，Anthropic出品） | [anthropic.com/research/building-effective-agents](https://www.anthropic.com/research/building-effective-agents) |

### ⭐⭐ 推荐

| 论文 | 年份 | 核心价值 |
|------|------|---------|
| Plan-then-Execute | 2025 | 规划/执行分离范式 |
| HiRAG: Hierarchical Knowledge RAG | 2025 | 分层知识索引 |
| GeAR: Graph-enhanced Agent for RAG | 2025 | 图增强Agent RAG |
| Agent-as-a-Judge | 2025 | 用Agent评估Agent |
| A Comprehensive Survey of Hallucination in LLMs | 2025 | 幻觉问题全景 |
| Not with a Prompt, but with a Handshake | 2025 | Prompt注入攻防全景 |

## 5.3 信息源

| 来源 | 频率 | 语言 | 价值 |
|------|------|------|------|
| [Lilian Weng Blog](https://lilianweng.github.io/) | 不定期 | 英文 | OpenAI研究负责人，质量极高 |
| [Anthropic Research](https://www.anthropic.com/research) | 不定期 | 英文 | Agent/安全/对齐 |
| [The Batch (Andrew Ng)](https://www.deeplearning.ai/the-batch/) | 周更 | 英文 | AI趋势周报 |
| [a16z AI Canon](https://a16z.com/ai-canon/) | 不定期 | 英文 | AI投资全景 |
| [LangChain Blog](https://blog.langchain.dev/) | 不定期 | 英文 | Agent/RAG工程 |
| [机器之心](https://www.jiqizhixin.com/) | 日更 | 中文 | 中文AI前沿 |
| [量子位](https://www.qbitai.com/) | 日更 | 中文 | 中文AI新闻 |

## 5.4 学习方法

| 方法 | 描述 | 适用场景 |
|------|------|---------|
| **费曼测试** | 能用一句话解释核心概念吗？ | 检验理解深度 |
| **论文阅读法** | 先读Abstract+Conclusion，再决定是否精读 | 高效筛选论文 |
| **项目驱动法** | 先定目标项目，倒推需要学什么 | 实战导向 |
| **500字笔记法** | 每完成一个模块，写500字总结 | 强制输出 |
| **Demo先行法** | 先跑通官方Demo，再改造成自己的场景 | 框架学习 |
| **红队思维法** | 对每个功能想"如果我是一个攻击者，怎么利用它？" | 安全意识培养 |

---

# 第六部分：深度专题

> 🆕 **v2.0 新增**: 本部分针对三大核心领域进行深度展开，帮助学习者在关键决策点做出正确选择。

## 6.1 Agent 架构选型指南

### 6.1.1 何时用 Agent，何时不用

> **核心原则**: Agent 带来了灵活性，但也带来了不确定性。能用 Workflow 解决的，就不要用 Agent。

| 场景 | 推荐方案 | 原因 |
|------|---------|------|
| 固定流程、步骤确定 | **Workflow（编排式）** | 可控、可测、可审计 |
| 用户意图多变性高 | **Agent（自主式）** | 灵活应对未知输入 |
| 需要调用外部工具 | **Agent + 工具** | 需要动态决策调用哪个工具 |
| 高风险操作（支付、删除） | **Workflow + 人类确认** | 安全第一 |
| 研究探索类任务 | **Agent** | 需要多步推理和迭代 |
| 数据ETL流水线 | **Workflow** | 确定性流程，无需推理 |

### 6.1.2 Agent 范式深度对比

| 维度 | ReAct | Plan-then-Execute | Reflexion |
|------|-------|-------------------|-----------|
| **决策模式** | 边想边做，每步推理 | 先规划全流程，再逐步执行 | 执行后反思，自我修正 |
| **优点** | 适应性强，单步灵活 | 全局视角，避免死循环 | 自我纠错能力 |
| **缺点** | 可能陷入循环 | 规划可能过时 | Token消耗高 |
| **适用场景** | 开放式问答、搜索 | 多步骤复杂任务 | 高精度要求的任务 |
| **Token消耗** | 中 | 低-中 | 高 |
| **可控性** | 低 | 高 | 中 |

### 6.1.3 框架选型决策树

```
你的需求是什么？
│
├── 快速原型 / MVP
│   └── → LangChain (最易上手，生态最全)
│
├── 复杂状态流转 / 多条件分支
│   └── → LangGraph (显式状态机，可视化)
│
├── 知识密集型 / RAG为主
│   └── → LlamaIndex (数据索引专精)
│
├── 多角色协作 / 分工明确
│   └── → CrewAI (角色定义直观)
│
├── 研究/实验/对话
│   └── → AutoGen (多Agent对话框架)
│
└── 生产级 / 高可靠性
    └── → LangGraph + 自定义节点 (完全可控)
```

### 6.1.4 生产环境 Agent 架构清单

| 能力 | 必要性 | 实现方式 |
|------|--------|---------|
| **错误处理** | 必须 | try-catch + 降级策略 + 重试 |
| **超时控制** | 必须 | 每个工具调用设置超时（≤30s） |
| **Token 预算** | 必须 | 单次对话 Token 上限 + 用量追踪 |
| **人类确认** | 高风险必须 | 高成本/不可逆操作前暂停 |
| **审计日志** | 必须 | 记录所有工具调用和决策 |
| **缓存** | 推荐 | 相同输入直接返回缓存结果 |
| **流式输出** | 推荐 | 长任务需流式返回中间结果 |
| **评估回路** | 推荐 | 自动评估输出质量 |

---

## 6.2 RAG 方案升级路线

### 6.2.1 四代 RAG 演化路径

```
第1代: Naive RAG
  文档→分块→向量化→检索→生成
  问题: 召回率低、幻觉多、无评估
  ↓
第2代: Advanced RAG
  + Query改写 + 混合检索 + Reranker + RAGAS评估
  问题: 无法回答跨文档的全局性问题
  ↓
第3代: GraphRAG
  + 知识图谱 + 社区检测 + 全局摘要
  问题: 构建成本高、更新困难
  ↓
第4代: Agentic RAG
  + Agent驱动 + 多步检索 + 自我反思 + 工具调用
  优势: 最灵活，能处理复杂问题
  代价: Token消耗高，延迟大
```

### 6.2.2 按场景选择 RAG 方案

| 场景 | 推荐方案 | 分块策略 | 向量库 | 检索策略 |
|------|---------|---------|--------|---------|
| **企业FAQ** | Naive RAG | 固定512字符 | Chroma | 纯向量 |
| **法律/合规文档** | Advanced RAG | 语义切分 | FAISS | 混合+Reranker |
| **研究论文库** | GraphRAG | 按章节 | Pinecone | 图+向量 |
| **客服知识库** | Advanced RAG | 递归切分 | Chroma | 混合+Reranker |
| **跨域分析报告** | Agentic RAG | 多粒度 | FAISS | Agent驱动多步 |

### 6.2.3 Chunking 策略深度对比

| 策略 | 原理 | 优点 | 缺点 | 最佳场景 |
|------|------|------|------|---------|
| **固定大小** | 每 N 个字符切一段 | 简单，可控 | 可能切断语义 | FAQ / 短文本 |
| **递归字符** | 按分隔符层级切分 | 保留段落结构 | 中文分隔符需调优 | 通用文档 |
| **语义切分** | 基于嵌入相似度判断边界 | 语义完整性最好 | 计算开销大 | 长文 / 学术 |
| **文档结构** | 按标题/章节切分 | 结构保留最好 | 需文档有清晰结构 | Markdown / HTML |
| **多粒度** | 同时生成粗/细两种粒度 | 兼顾全局和细节 | 存储翻倍 | GraphRAG / 复杂查询 |

### 6.2.4 RAG 评估指标详解

| 指标 | 含义 | 计算方式 | 目标值 |
|------|------|---------|--------|
| **Faithfulness** | 回答是否忠实于上下文 | 可被上下文支撑的claim数 / 总claim数 | ≥ 0.85 |
| **Answer Relevancy** | 回答与问题的相关度 | 问题→回答→生成问题的相似度 | ≥ 0.80 |
| **Context Precision** | 检索结果中有用内容的比例 | 相关chunk排名 / 总chunk数 | ≥ 0.75 |
| **Context Recall** | 答案所需信息被检索到的比例 | 被检索到的GT句子 / GT总句子 | ≥ 0.80 |
| **Answer Correctness** | 回答的正确性 | 与标准答案的语义相似度 | ≥ 0.85 |

```python
# RAGAS 评估示例
# pip install ragas

from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)
from datasets import Dataset

# 准备评估数据
eval_data = {
    "question": ["文档的主要观点是什么？"],
    "answer": ["文档主要讨论了..."],
    "contexts": [["检索到的上下文片段1", "片段2"]],
    "ground_truth": ["标准答案：文档主要讨论了..."],
}

dataset = Dataset.from_dict(eval_data)
result = evaluate(
    dataset,
    metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
)
print(result)
```

---

## 6.3 12 周路线定制化策略

### 6.3.1 按角色定制学习路线

| 角色 | 侧重阶段 | 时间分配调整 | 跳过/深化的模块 |
|------|---------|-------------|---------------|
| **AI 工程师** | 阶段一+二 (80%) | 实践60% | 深化Agent+RAG，跳过商业层 |
| **AI 产品经理** | 阶段一(30%)+三(70%) | 理论40%+案例40% | 跳过代码实现，深化产品+商业 |
| **技术决策者** | 全阶段均匀 | 理论50%+案例50% | 跳过手写代码，深化架构选型+评估 |
| **数据科学家** | 阶段一(60%)+二(40%) | 理论50%+实践40% | 深化ML/DL，适度接触Agent |
| **转行者** | 全阶段，节奏放慢 | 每周7h→14周 | 不跳过，增加补基础时间 |

### 6.3.2 按已有基础定制

| 已有基础 | 可跳过 | 可加速 | 需额外投入 |
|----------|--------|--------|-----------|
| Python 熟练 | 第1周环境搭建 | 第1周直接进AI工具链 | 无 |
| ML 经验 | 第2周全部 | 第3周CNN部分 | Transformer深入 |
| DL 经验 | 第2-3周 | 第4周LLM架构 | RAG + Agent |
| LLM 应用经验 | 第1-4周 | 第5-6周RAG基础 | GraphRAG + 安全 |
| 全栈工程师 | 第1周 | 部署相关 | 数学补强 |

### 6.3.3 学习节奏调整方案

| 节奏 | 周学时 | 总周期 | 适合人群 | 风险 |
|------|--------|--------|---------|------|
| **紧凑型** | 15h | 8周 | 全职学习者 | 消化不足 |
| **标准型** | 10h | 12周 | 在职学习者 | 无 |
| **宽松型** | 5h | 24周 | 时间有限者 | 动力维持难 |
| **周末型** | 8h (仅周末) | 15周 | 周末集中学 | 平日遗忘 |

### 6.3.4 每周自检清单

> 完成每周学习后，用以下清单自检。**任意一项为"否"都应回头补强。**

| 检查项 | 通过标准 |
|--------|---------|
| 我能用自己的话解释本周核心概念吗？ | 费曼测试通过 |
| 本周的代码/项目能独立跑通吗？ | 无需查资料 |
| 我能指出本周内容的一个局限或争议吗？ | 批判性思维 |
| 本周的产出物已保存到GitHub/笔记了吗？ | 可回溯 |
| 我能回答"这周学了什么，为什么学"吗？ | 目标感 |

---

# 附录

## A. 术语表

| 术语 | 英文 | 定义 |
|------|------|------|
| 大语言模型 | LLM (Large Language Model) | 基于Transformer的大规模预训练语言模型 |
| 检索增强生成 | RAG (Retrieval-Augmented Generation) | 结合外部知识检索的文本生成技术 |
| 智能体 | Agent | 能感知环境并自主执行动作的AI系统 |
| 提示工程 | Prompt Engineering | 设计优化输入提示以引导LLM输出的技术 |
| 微调 | Fine-tuning | 在预训练模型基础上用特定数据继续训练 |
| 向量化 | Embedding | 将文本转化为数值向量表示 |
| 幻觉 | Hallucination | 模型生成看似合理但实际错误的内容 |
| 对齐 | Alignment | 使模型行为符合人类意图和价值观 |
| 红队测试 | Red Teaming | 模拟攻击者对AI系统进行安全测试 |
| 护栏 | Guardrails | 输入/输出过滤层，阻止不安全内容 |
| Prompt 注入 | Prompt Injection | 通过恶意输入覆盖系统指令的攻击方式 |
| 越狱 | Jailbreak | 绕过模型安全限制获取被禁止输出的技术 |

## B. 推荐书籍

| 书名 | 作者 | 适合阶段 |
|------|------|---------|
| 《动手学深度学习》 | 李沐 等 | 阶段一 |
| 《机器学习》(西瓜书) | 周志华 | 阶段一 |
| 《Deep Learning》 | Goodfellow 等 | 阶段一~二 |
| 《Co-Intelligence》 | Ethan Mollick | 阶段三 |
| 《The Coming Wave》 | Mustafa Suleyman | 阶段三 |
| 《Designing Machine Learning Systems》 | Chip Huyen | 阶段二~三 |

## C. 认证路径

| 认证 | 提供方 | 费用 | 适合人群 |
|------|--------|------|---------|
| AWS ML Specialty | Amazon | $300 | 云端ML工程师 |
| Google Professional ML Engineer | Google | $200 | GCP用户 |
| Azure AI Fundamentals (AI-900) | Microsoft | $99 | 企业AI入门 |
| TensorFlow Developer Certificate | Google | $100 | TF开发者 |
| DeepLearning.AI Certificates | DeepLearning.AI | 免费 | 通用 |

## D. AI 安全工具速查

| 工具 | 类型 | 用途 | 链接 |
|------|------|------|------|
| Garak | 红队测试 | LLM漏洞扫描 | [github.com/NVIDIA/garak](https://github.com/NVIDIA/garak) |
| NeMo Guardrails | 护栏 | 输入/输出安全过滤 | [github.com/NVIDIA/NeMo-Guardrails](https://github.com/NVIDIA/NeMo-Guardrails) |
| PromptFuzz | 模糊测试 | Prompt注入检测 | [github.com/promptfuzz](https://github.com/promptfuzz) |
| AIRTBench | 评估 | AI安全基准测试 | [airtbench.com](https://airtbench.com) |
| LLM Guard | 安全 | 输入/输出扫描 | [github.com/protectai/llm-guard](https://github.com/protectai/llm-guard) |

---

> **下一步建议**: 从第0周的预算规划开始，确认环境和资金就绪后，按标准节奏（每周10小时）推进12周。记住三句话：**先跑通，再理解，持续输出；安全思维从第一天养成；用评估驱动迭代，不是凭感觉。**

---

*v2.0 | 本教程基于对全球Top 10 AI课程、中文生态Top 5教程、37篇前沿论文的系统分析生成。v2.0新增：第0周预算规划、AI安全模块、RAG代码升级、毕业评估矩阵、Agent架构选型指南、RAG升级路线、12周定制化策略。*
