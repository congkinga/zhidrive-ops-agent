# 智驾产品体验运营增长实验

这是一个面向智能驾驶产品运营场景的产品原型，基于开源项目 [tsingyuai/growth-lab](https://github.com/tsingyuai/growth-lab) 的“观察、行动、复盘”框架实现。

## 项目解决的问题

智驾体验反馈经常停留在“不好用、有点慌、变道犹豫”等主观描述里。这个项目把它转成可定位、可派发、可验证、可复盘的标准化运营任务。

## 当前产物

- 产品理解：`SOUL.md`
- 静态 Demo：`product/landing/zhijia-nova-ops.html`
- 聊天 Agent 原型：`product/landing/zhijia-agent.html`
- RAG 检索引擎：`rag_engine.py`
- 公开行业观察：`docs/zhijia-industry-observation.md`
- SEO 调研与页面审阅：`memory/run-seo-page-loop/`
- Agent 架构说明：`docs/agent-architecture.md`
- 项目总说明：`CASE_STUDY.md`

## 快速运行

启动 Agent 服务：

```powershell
& '.\.venv\Scripts\python.exe' agent_server.py
```

然后访问：

`http://127.0.0.1:8765/product/landing/zhijia-agent.html`

也可以直接打开静态表单：

`product/landing/zhijia-nova-ops.html`

也可以在本目录启动静态服务器：

```powershell
python -m http.server 8080
```

然后访问 `http://localhost:8080/product/landing/zhijia-nova-ops.html`

## 可选大模型配置

不配置 API Key 时，Agent 使用本地文档检索和规则回答。

推荐在项目根目录创建 `.env.local`：

```powershell
OPENAI_API_KEY=your-key
OPENAI_MODEL=gpt-4.1-mini
```

DeepSeek：

```powershell
DEEPSEEK_API_KEY=your-key
DEEPSEEK_MODEL=deepseek-chat
```

也可以直接在启动服务前设置环境变量：

```powershell
$env:OPENAI_API_KEY="your-key"
python agent_server.py
```

不要把 Key 写进 `agent_server.py`、HTML 或任何会提交到 Git 的文件。

## 核心流程

```text
场景录入
-> 现象与证据整理
-> 五维问题分类
-> 严重度标记与责任方建议
-> 短期动作
-> 验证方式
-> 复盘口径
```

## 智能解析

Agent 支持把自然语言体验反馈自动解析为结构化问题记录：

```text
用户反馈
-> RAG 检索相关知识
-> DeepSeek 抽取场景、严重度、问题维度、责任方
-> 输出可验证的结构化结果
```

## 系统能力

- RAG 混合检索：TF-IDF + BM25
- DeepSeek / OpenAI 可选大模型
- 自然语言体验反馈结构化分析
- 多轮会话上下文
- 案例库与评估集
- 模型调用日志与系统指标
- Markdown 运行报告

## 边界

- 不包含博世内部数据
- 不公开未确认的具体车型缺陷
- 不把公开行业趋势写成个人实测结论
- 当前 Demo 不连接后端，不保存输入

## 上游说明

本项目基于 Growth Lab 的框架和目录结构进行二次开发。上游 README 保留在 `UPSTREAM_GROWTH_LAB.md`。
