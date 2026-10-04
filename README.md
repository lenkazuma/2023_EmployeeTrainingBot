# 员工培训手册问答机器人 · LangChain + 百度千帆

基于 Streamlit、LangChain 和百度千帆 ERNIE 大模型的文档问答机器人：加载一份培训手册或公司文档（PDF），自动生成摘要，员工可以连续提问，回答只依据文档内容并注明参考页码，结束时还能生成对话总结。

A Streamlit chatbot that answers employees' questions about a training manual (or any PDF) using LangChain and Baidu Qianfan ERNIE models, with page references and conversation summaries.

## 功能

- **多种文档来源**：内置示例（万科 2023 年一季报），也可以上传自己的 PDF 或填写 PDF 链接。
- **文档摘要**：加载后自动生成中文摘要。
- **带记忆的问答**：聊天界面，记住最近几轮对话，回答注明参考页码；文档中找不到时会直接说明。
- **对话总结**：点击「结束对话并总结」生成本次问答要点。
- **索引缓存**：按文档内容哈希把向量索引保存在 `.index_cache/`，同一份文档只需计算一次 embedding；换文档不会误用旧索引。
- **模型可选**：ERNIE-3.5-8K / ERNIE-4.0-8K / ERNIE-Speed-8K / ERNIE-Lite-8K。

## 安装

需要 Python 3.10+。

```bash
git clone https://github.com/lenkazuma/EmployeeTrainingBot.git
cd EmployeeTrainingBot
pip install -r requirements.txt
```

## 配置千帆凭证

在 [百度智能云千帆控制台](https://console.bce.baidu.com/qianfan/) 创建应用获取 API Key / Secret Key，然后任选一种方式：

- 在项目根目录创建 `.env`：

  ```env
  QIANFAN_AK=your-api-key
  QIANFAN_SK=your-secret-key
  ```

- 或运行后在页面侧边栏填写（只保存在当前会话中）。

## 运行

```bash
streamlit run streamlit_app.py
```

浏览器打开 <http://localhost:8501>，在侧边栏选择文档来源后即可提问。

## 项目结构

```
├── streamlit_app.py   # Streamlit 界面
├── bot_core.py        # PDF 解析、切块、索引缓存、问答与摘要
└── tests/             # pytest 测试（不需要千帆凭证）
```

## 测试

```bash
pip install pytest
pytest -q
```

## 许可证

本项目基于 MIT 许可证授权，详见 [LICENSE](LICENSE)。
