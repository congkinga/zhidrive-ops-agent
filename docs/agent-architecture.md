# 智驾运营 Agent 架构说明

## 当前原型

当前版本由两部分组成：

- 前端：`product/landing/zhijia-agent.html`
- 本地服务：`agent_server.py`

启动服务：

```powershell
& '.\.venv\Scripts\python.exe' agent_server.py
```

访问：

```text
http://127.0.0.1:8765/product/landing/zhijia-agent.html
```

服务会读取项目文档并建立本地 RAG 索引。当前检索由 `rag_engine.py` 完成：

- Markdown 按标题和段落切块
- 中文单字、中文二元组和英文词构建 TF-IDF 向量
- 同时计算 BM25
- 向量相似度和 BM25 按 0.72 / 0.28 混合排序
- 把最相关片段交给 DeepSeek 或 OpenAI 生成回答
- 配置 `OPENAI_EMBEDDING_MODEL` 后，可使用 OpenAI embedding 替代本地 TF-IDF 向量

没有配置大模型 API Key 时，使用本地检索和规则回答；配置 API Key 后，自动切换到大模型回答。

能力包括：

- 回答项目说明、操作方法和五维分类
- 解释行业观察和体验闭环方法
- 通过多轮问题引导用户录入体验问题
- 生成标准化的体验问题记录
- 将自然语言体验反馈自动解析为结构化问题
- 保留多轮会话上下文
- 自动保存分析案例
- 运行评估集并计算结构化抽取准确率
- 记录模型调用耗时与来源
- 从项目文档中检索相关内容作为回答依据

## 为什么保留本地回退

本地规则回退适合产品原型验证，因为：

- 不依赖 API Key，环境不可用时仍然可以演示
- 可以清楚展示产品交互和多轮流程
- 即使没有后端，也能验证信息收集和闭环模板

## 真实 Agent 的升级路径

如果要把这个 Demo 变成真实可用 Agent，可以按以下路径做：

### 1. 接入大模型

当前服务已支持 OpenAI 和 DeepSeek 的 OpenAI 兼容接口。

OpenAI：

```powershell
python agent_server.py
```

DeepSeek：

```powershell
python agent_server.py
```

推荐把 Key 放在项目根目录的 `.env.local` 文件里，不要写进 `agent_server.py` 或前端 HTML。

`.env.local` 示例：

```text
OPENAI_API_KEY=your-key
OPENAI_MODEL=gpt-4.1-mini
```

或者：

```text
DEEPSEEK_API_KEY=your-key
DEEPSEEK_MODEL=deepseek-chat
```

该文件已被 `.gitignore` 排除，不会进入 Git 提交。

也可以设置兼容接口：

```powershell
$env:OPENAI_BASE_URL="https://your-compatible-endpoint/v1"
```

没有配置 Key 时，服务会自动使用本地规则和文档检索。

### 2. 使用知识库检索

当前服务已经把以下文件变成本地知识库：

- `SOUL.md`
- `docs/zhijia-industry-observation.md`
- `CASE_STUDY.md`
- `product/README.md`

用户提问时，先从知识库检索相关内容，再把检索结果作为上下文交给大模型。

也可以直接查看检索结果：

```powershell
Invoke-RestMethod `
  -Uri 'http://127.0.0.1:8765/api/rag/search' `
  -Method Post `
  -ContentType 'application/json; charset=utf-8' `
  -Body '{"message":"NOA 场景评测","top_k":5}'
```

也可以调用结构化分析接口：

```powershell
Invoke-RestMethod `
  -Uri 'http://127.0.0.1:8765/api/analyze' `
  -Method Post `
  -ContentType 'application/json; charset=utf-8' `
  -Body '{"text":"城市道路无保护左转时系统犹豫，进入路口后 NOA 降级，接管提醒较晚。"}'
```

### 3. 使用工具调用

当前原型已经支持：

- 读取本地项目文档
- 通过 `/api/chat` 回答问题
- 通过 `/api/analyze` 自动分析体验反馈
- 通过 `/api/cases` 读取案例
- 通过 `/api/eval` 运行评估
- 通过 `/api/report` 生成报告
- 通过 `/api/metrics` 查看系统指标
- 在没有大模型时回退到本地规则

后续可以继续增加：

- 调用公开搜索或行业数据接口
- 自动执行更复杂的跨工具任务

### 4. 结构化输出

要求模型输出 JSON，例如：

```json
{
  "scenario": "城市道路",
  "observation": "无保护左转时系统犹豫，进入路口后 NOA 降级",
  "severity": "A",
  "dimensions": ["决策", "交互", "场景边界"],
  "owner": "规划控制、HMI、产品运营联合定位",
  "short_term_action": "补充视频和车端日志，确认触发条件",
  "verification": "同场景复测，观察是否稳定复现"
}
```

## 与项目关系

这个 Agent 可以把原来的静态表单变成自然语言入口，进一步体现 AI 产品运营中的交互设计、提示词设计、知识库管理和结构化输出能力。
