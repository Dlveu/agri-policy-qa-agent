"""
农业政策智能问答 Agent
增强功能：
- 多轮动态追问引导（根据意图只追问必要信息）
- Streamlit Web可视化界面（替代CLI，更友好）
- 原有核心功能全部保留
"""

import os
import re
from typing import List, Optional, Literal, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime
import streamlit as st

# LangChain 相关导入
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    AIMessage,
    SystemMessage
)
from langchain_core.prompts import PromptTemplate
from langgraph.graph import StateGraph, END
from langchain_community.vectorstores.faiss import FAISS
import dotenv

# =========================
# 环境变量加载 & 配置项
# =========================
dotenv.load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")
if not OPENAI_API_KEY:
    raise EnvironmentError("未检测到 OPENAI_API_KEY 环境变量！")

# 常量定义
LLM_MODEL = "gpt-4o-mini"
LLM_TEMPERATURE = 0.2
RAG_TOP_K = 3
FAISS_INDEX_PATH = "faiss_index"

# 记忆配置
SHORT_MEMORY_TOP_K = 5
SUMMARY_TRIGGER_ROUNDS = 3

# 通用意图关键词
GREETING_KEYWORDS = ["你好", "您好", "嗨", "哈喽", "早上好", "下午好", "晚上好"]
THANKS_KEYWORDS = ["谢谢", "感谢", "多谢", "辛苦了"]
FAREWELL_KEYWORDS = ["再见", "拜拜", "下次见", "回见"]
IDENTITY_KEYWORDS = ["你是谁", "你叫什么", "名字", "身份"]
FUNCTION_KEYWORDS = ["你能做什么", "功能", "能干什么", "帮助", "作用"]
GENERAL_KEYWORDS = GREETING_KEYWORDS + THANKS_KEYWORDS + FAREWELL_KEYWORDS + IDENTITY_KEYWORDS + FUNCTION_KEYWORDS

# 长记忆摘要提示词
SUMMARY_PROMPT = """
请总结以下农业政策问答对话的核心信息，要求：
1. 保留关键信息：用户关注的地区、作物、政策类型、核心问题
2. 去除冗余内容，只保留有价值的信息
3. 格式简洁，使用要点式总结
4. 忽略无关的寒暄内容

对话历史：
{conversation_history}

当前时间：{current_time}

总结要求：仅输出总结内容，不要额外解释
"""

# =========================
# 工具函数
# =========================
def trim_short_memory(messages: List[BaseMessage], top_k: int = SHORT_MEMORY_TOP_K) -> List[BaseMessage]:
    """手动修剪短记忆，兼容所有 LangChain 版本"""
    if not messages:
        return []
    system_messages = [msg for msg in messages if isinstance(msg, SystemMessage)]
    conversation_messages = [msg for msg in messages if not isinstance(msg, SystemMessage)]
    keep_count = top_k * 2
    trimmed_conversation = conversation_messages if len(conversation_messages) <= keep_count else conversation_messages[-keep_count:]
    return system_messages + trimmed_conversation

def generate_long_memory_summary(messages: List[BaseMessage], llm: ChatOpenAI) -> str:
    """生成长记忆摘要"""
    conv_history = ""
    for msg in messages:
        if isinstance(msg, HumanMessage):
            conv_history += f"用户：{msg.content}\n"
        elif isinstance(msg, AIMessage):
            conv_history += f"AI：{msg.content}\n"
    prompt = PromptTemplate(template=SUMMARY_PROMPT, input_variables=["conversation_history", "current_time"])
    summary_input = prompt.format(
        conversation_history=conv_history,
        current_time=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    )
    response = llm.invoke([HumanMessage(content=summary_input)])
    return response.content.strip()

# =========================
# 数据模型定义
# =========================
class AgentState(BaseModel):
    messages: List[BaseMessage] = Field(default_factory=list)
    user_question: Optional[str] = None
    intent_type: Optional[Literal[
        "greeting",          # 问候
        "thanks",            # 感谢
        "farewell",          # 告别
        "identity",          # 身份询问
        "function",          # 功能询问
        "policy_explanation",# 政策解读
        "eligibility_check", # 资格核查
        "calculation",       # 金额计算
        "procedure",         # 办理流程
        "unclear"            # 意图不明
    ]] = None
    short_term_facts: Dict[str, Any] = Field(default_factory=dict)
    long_term_profile: Dict[str, Any] = Field(default_factory=lambda: {"summary": "", "conversation_round": 0})
    need_rag: bool = False
    retrieved_docs: List[Dict[str, Any]] = Field(default_factory=list)
    need_clarification: bool = False
    refuse_answer: bool = False
    final_answer: Optional[str] = None

# =========================
# LangGraph 节点函数
# =========================
def parse_user_input(state: AgentState) -> AgentState:
    """解析用户输入，提取问题"""
    for msg in reversed(state.messages):
        if isinstance(msg, HumanMessage):
            state.user_question = msg.content.strip()
            break
    state.long_term_profile["conversation_round"] = state.long_term_profile.get("conversation_round", 0) + 1
    state.messages = trim_short_memory(state.messages, SHORT_MEMORY_TOP_K)
    return state

def classify_intent(state: AgentState) -> AgentState:
    """意图分类节点：优先识别所有通用话术，再识别政策意图"""
    user_question = state.user_question or ""

    # 通用意图判断
    if any(word in user_question for word in GREETING_KEYWORDS):
        state.intent_type = "greeting"
        state.need_rag = False
        state.need_clarification = False
        return state
    elif any(word in user_question for word in THANKS_KEYWORDS):
        state.intent_type = "thanks"
        state.need_rag = False
        state.need_clarification = False
        return state
    elif any(word in user_question for word in FAREWELL_KEYWORDS):
        state.intent_type = "farewell"
        state.need_rag = False
        state.need_clarification = False
        return state
    elif any(word in user_question for word in IDENTITY_KEYWORDS):
        state.intent_type = "identity"
        state.need_rag = False
        state.need_clarification = False
        return state
    elif any(word in user_question for word in FUNCTION_KEYWORDS):
        state.intent_type = "function"
        state.need_rag = False
        state.need_clarification = False
        return state

    # 政策相关意图判断
    if any(keyword in user_question for keyword in ["补贴", "政策", "规定", "文件"]):
        state.intent_type = "policy_explanation"
        state.need_rag = True
    elif any(keyword in user_question for keyword in ["能不能", "符合", "资格", "条件"]):
        state.intent_type = "eligibility_check"
        state.need_rag = True
    elif any(keyword in user_question for keyword in ["多少钱", "怎么算", "金额", "标准"]):
        state.intent_type = "calculation"
        state.need_rag = True
    elif any(keyword in user_question for keyword in ["去哪", "什么时候", "怎么申请", "流程"]):
        state.intent_type = "procedure"
        state.need_rag = True
    else:
        state.intent_type = "unclear"
        state.need_clarification = True

    return state

def general_response_node(state: AgentState) -> AgentState:
    """通用回复节点，处理问候/感谢/告别/身份/功能询问"""
    intent = state.intent_type
    responses = {
        "greeting": "您好！我是农业政策智能问答助手，请问有什么可以帮您解答的农业政策问题吗？😊",
        "thanks": "不客气！如果您还有其他农业政策相关的问题，随时可以问我哦~",
        "farewell": "再见！感谢您的使用，祝您生活愉快！👋",
        "identity": "我是农业政策智能问答助手，专为您解答各类农业相关的政策问题，比如补贴标准、申请流程、资格条件等~",
        "function": "我可以帮您解答农业政策相关的各类问题，包括：\n1. 各类农业补贴的标准和申请条件\n2. 农业项目的申报流程\n3. 相关政策文件的解读\n您可以直接告诉我您想了解的内容~"
    }
    state.final_answer = responses.get(intent, "很高兴为您服务！")
    state.messages.append(AIMessage(content=state.final_answer))
    return state

def clarification_node(state: AgentState) -> AgentState:
    """
    核心优化：多轮动态追问引导
    根据不同意图，只追问必要的信息，而非一次性列出所有要求
    """
    user_question = state.user_question or ""
    intent = state.intent_type
    long_memory = state.long_term_profile.get("summary", "")

    # 从历史对话/当前问题中提取已有的信息
    has_region = bool(re.search(r"[省市县]", user_question)) or ("地区" in long_memory)
    has_crop = bool(re.search(r"小麦|玉米|水稻|蔬菜|大豆", user_question)) or ("作物" in long_memory)
    has_year = bool(re.search(r"20\d{2}", user_question)) or ("年份" in long_memory)

    # 按意图动态生成追问话术
    clarify_map = {
        # 政策解读：优先追问地区（核心）
        "policy_explanation":
            "请问您想查询哪个省/市/县的政策呢？" if not has_region else
            ("请问您关注的是哪一年的政策（如2025）？" if not has_year else
             "请问您想了解哪种作物的政策（如小麦、玉米）？" if not has_crop else
             "请补充您想了解的具体方向（如补贴标准、申请条件）~"),

        # 资格核查：优先追问地区+作物
        "eligibility_check":
            "请问您的种植地区是哪个省/市/县，种植的是什么作物呢？" if not has_region or not has_crop else
            ("请问您想查询哪一年的资格条件？" if not has_year else
             "请补充更多信息（如种植面积、是否符合基本条件）~"),

        # 金额计算：优先追问地区+作物+面积
        "calculation":
            "请问您的种植地区、作物类型和种植面积分别是多少呢？" if not has_region or not has_crop else
            ("请问您想按哪一年的补贴标准计算？" if not has_year else
             "请补充种植面积，我来帮您计算补贴总额~"),

        # 流程查询：优先追问地区
        "procedure":
            "请问您所在的地区是哪个省/市/县？" if not has_region else
            ("请问您想了解哪一年的申请流程？" if not has_year else
             "请问您想了解哪种作物的申请流程？" if not has_crop else
             "请补充您想了解的具体流程环节（如申报时间、所需材料）~"),

        # 完全不明的问题：简化追问
        "unclear": "为了精准回答您的问题，请补充：\n1. 所在地区（省/市/县）\n2. 涉及的作物类型（如小麦、玉米）"
    }

    # 生成最终追问话术
    clarify_msg = clarify_map.get(intent, clarify_map["unclear"])
    # 补充记忆上下文（如果有）
    if long_memory and intent != "unclear":
        clarify_msg = f"根据之前的对话，{clarify_msg}"

    state.final_answer = clarify_msg
    state.messages.append(AIMessage(content=clarify_msg))
    return state

def rag_retrieval_node(state: AgentState, vectorstore: FAISS) -> AgentState:
    """RAG 检索节点"""
    if state.need_rag and state.user_question:
        try:
            retrieved_documents = vectorstore.similarity_search(state.user_question, k=RAG_TOP_K)
            state.retrieved_docs = [
                {"page_content": doc.page_content.strip(), "source": doc.metadata.get("source", "未知文件")}
                for doc in retrieved_documents
            ]
        except Exception as e:
            print(f"RAG 检索出错: {e}")
            state.retrieved_docs = []
    return state

def llm_expert_answer(state: AgentState) -> AgentState:
    """政策回答节点"""
    long_memory = state.long_term_profile.get("summary", "")
    memory_context = f"\n【对话历史总结】：{long_memory}\n" if long_memory else ""

    system_prompt = f"""
你是农业农村政策专家，请【严格遵守】以下规则：
1. 只能基于【政策原文证据】和【对话历史】回答
2. 每一个结论，必须先引用原文句子，格式为：
   【政策原文】……
   【通俗解读】……
3. 无明确依据时，回答：未在已检索政策中找到明确依据，建议咨询当地农业农村局
4. 禁止自行推断、补全常识、编造内容

【对话上下文】
{memory_context}

回答语言要朴实，面向农户。
"""

    # 构造证据
    evidence_blocks = ""
    if state.retrieved_docs:
        aggregated = aggregate_sentences(state.retrieved_docs)
        evidence_blocks = "\n【可用政策证据】\n"
        for i, item in enumerate(aggregated, 1):
            evidence_blocks += f"\n【证据{i}｜来源：{item['source']}】\n{item['content']}\n"

    # 调用LLM
    llm = ChatOpenAI(
        model=LLM_MODEL,
        temperature=0.1,
        api_key=OPENAI_API_KEY,
        base_url=OPENAI_BASE_URL
    )
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=evidence_blocks),
        HumanMessage(content=f"用户问题：{state.user_question}")
    ]
    response = llm.invoke(messages)

    state.final_answer = response.content
    state.messages.append(AIMessage(content=response.content))
    return state

def update_long_memory(state: AgentState) -> AgentState:
    """更新长记忆节点"""
    current_round = state.long_term_profile.get("conversation_round", 0)

    if current_round % SUMMARY_TRIGGER_ROUNDS == 0 and current_round > 0:
        # Streamlit中用st.info替代print
        st.info(f"🔍 正在更新对话记忆（第 {current_round} 轮）...")

        llm = ChatOpenAI(
            model=LLM_MODEL,
            temperature=0.1,
            api_key=OPENAI_API_KEY,
            base_url=OPENAI_BASE_URL
        )
        new_summary = generate_long_memory_summary(state.messages, llm)

        # 合并新旧摘要
        old_summary = state.long_term_profile.get("summary", "")
        if old_summary:
            state.long_term_profile["summary"] = f"历史总结：{old_summary}\n最新总结：{new_summary}"
        else:
            state.long_term_profile["summary"] = new_summary

        st.success(f"📝 记忆更新完成：{state.long_term_profile['summary'][:100]}...")

    return state

def aggregate_sentences(docs: List[Dict[str, Any]], window: int = 1) -> List[Dict[str, Any]]:
    """聚合命中句子为弱段落"""
    aggregated = []
    for i, doc in enumerate(docs):
        sentences = [doc["page_content"]]
        if i - window >= 0:
            sentences.insert(0, docs[i - window]["page_content"])
        if i + window < len(docs):
            sentences.append(docs[i + window]["page_content"])
        aggregated.append({
            "content": "\n".join(sentences),
            "evidence": doc["page_content"],
            "source": doc.get("source", "未知文件")
        })
    return aggregated

# =========================
# 构建 LangGraph 工作流
# =========================
def build_agricultural_policy_agent(vectorstore: FAISS):
    """构建带记忆和增强通用能力的Agent"""
    workflow = StateGraph(AgentState)

    # 添加节点
    workflow.add_node("parse_input", parse_user_input)
    workflow.add_node("classify_intent", classify_intent)
    workflow.add_node("general_response", general_response_node)
    workflow.add_node("update_long_memory", update_long_memory)
    workflow.add_node("clarify", clarification_node)
    workflow.add_node("rag_retrieval", lambda s: rag_retrieval_node(s, vectorstore))
    workflow.add_node("generate_answer", llm_expert_answer)

    # 设置入口节点
    workflow.set_entry_point("parse_input")

    # 定义执行流程
    workflow.add_edge("parse_input", "classify_intent")

    # 意图路由函数
    def route_intent(state: AgentState) -> str:
        if state.intent_type in ["greeting", "thanks", "farewell", "identity", "function"]:
            return "general_response"
        elif state.need_clarification:
            return "clarify"
        else:
            return "rag_retrieval"

    # 条件分支
    workflow.add_conditional_edges(
        source="classify_intent",
        path=route_intent,
        path_map={
            "general_response": "general_response",
            "clarify": "clarify",
            "rag_retrieval": "rag_retrieval"
        }
    )

    # 后续流程
    workflow.add_edge("general_response", "update_long_memory")
    workflow.add_edge("rag_retrieval", "generate_answer")
    workflow.add_edge("generate_answer", "update_long_memory")
    workflow.add_edge("clarify", "update_long_memory")
    workflow.add_edge("update_long_memory", END)

    return workflow.compile()

# =========================
# Streamlit Web界面（核心新增）
# =========================
def streamlit_chat_interface():
    """
    替代CLI的Web可视化界面
    - 友好的对话界面
    - 保存对话历史
    - 适配多轮追问
    """
    # 页面基础配置
    st.set_page_config(
        page_title="农业政策智能问答助手",
        page_icon="🌾",
        layout="wide"
    )

    st.title("🌾 农业政策智能问答助手")
    st.markdown("### 专注解答各类农业政策问题（补贴、申请、资格等）")
    st.divider()

    # 1. 初始化FAISS向量库（只加载一次）
    @st.cache_resource
    def load_vector_store():
        try:
            embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)
            vector_store = FAISS.load_local(
                folder_path=FAISS_INDEX_PATH,
                embeddings=embeddings,
                allow_dangerous_deserialization=True
            )
            return vector_store
        except Exception as e:
            st.error(f"加载政策知识库失败：{e}")
            st.stop()

    # 2. 初始化Agent（只加载一次）
    @st.cache_resource
    def load_agent():
        vector_store = load_vector_store()
        return build_agricultural_policy_agent(vector_store)

    # 3. 初始化会话状态（保存对话历史和Agent状态）
    if "agent_state" not in st.session_state:
        st.session_state.agent_state = AgentState(messages=[])
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # 加载资源
    vector_store = load_vector_store()
    policy_agent = load_agent()

    # 显示历史对话
    for msg in st.session_state.chat_history:
        if msg["role"] == "user":
            with st.chat_message("user"):
                st.markdown(msg["content"])
        else:
            with st.chat_message("assistant"):
                st.markdown(msg["content"])

    # 用户输入框
    user_input = st.chat_input("请输入您的问题（如：北京市2025年小麦补贴多少？）")
    if user_input:
        # 显示用户消息
        with st.chat_message("user"):
            st.markdown(user_input)
        st.session_state.chat_history.append({"role": "user", "content": user_input})

        # 调用Agent处理
        try:
            # 更新Agent状态
            st.session_state.agent_state.messages.append(HumanMessage(content=user_input))
            # 执行Agent工作流
            result = policy_agent.invoke(st.session_state.agent_state)
            # 转换回AgentState对象
            if isinstance(result, dict):
                st.session_state.agent_state = AgentState(**result)
            else:
                st.session_state.agent_state = result

            # 获取回答并显示
            answer = st.session_state.agent_state.final_answer
            with st.chat_message("assistant"):
                st.markdown(answer)
            st.session_state.chat_history.append({"role": "assistant", "content": answer})

        except Exception as e:
            st.error(f"回答生成出错：{str(e)}")
            # 回滚状态
            st.session_state.agent_state.messages.pop()

    # 侧边栏：重置对话
    with st.sidebar:
        st.header("⚙️ 功能设置")
        if st.button("清空对话历史", type="secondary"):
            st.session_state.agent_state = AgentState(messages=[])
            st.session_state.chat_history = []
            st.rerun()
        st.markdown("---")
        st.markdown("### 使用说明：")
        st.markdown("1. 支持查询各地区农业补贴政策")
        st.markdown("2. 支持询问补贴申请流程、资格条件")
        st.markdown("3. 支持计算补贴金额")
        st.markdown("4. 支持多轮追问，逐步补充信息")

# =========================
# 兼容CLI模式（保留原有功能）
# =========================
def interactive_chat(agent, vector_store):
    """原有的CLI交互模式，备用"""
    print("="*60)
    print("      农业政策智能问答助手（输入 'exit' 退出）")
    print("="*60)

    current_state = AgentState(messages=[])

    while True:
        user_input = input("\n👉 请输入您的问题：").strip()

        if user_input.lower() in ["exit", "quit", "退出", "结束"]:
            print("\n👋 感谢使用，再见！")
            break

        if not user_input:
            print("⚠️  请输入有效的问题！")
            continue

        current_state.messages.append(HumanMessage(content=user_input))

        try:
            result = agent.invoke(current_state)
            if isinstance(result, dict):
                current_state = AgentState(**result)
            else:
                current_state = result

            print("\n🤖 回答：")
            print(current_state.final_answer)

        except Exception as e:
            print(f"\n❌ 回答生成出错：{e}")
            import traceback
            traceback.print_exc()
            current_state.messages.pop()

# =========================
# 主程序入口
# =========================
if __name__ == "__main__":
    # 默认启动Streamlit Web界面
    try:
        streamlit_chat_interface()
    # 如果环境不支持Streamlit（如无Web环境），自动降级到CLI模式
    except Exception as e:
        print(f"启动Web界面失败，切换到CLI模式：{e}")
        # 加载向量库
        print("正在加载FAISS向量库...")
        try:
            embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)
            vector_store = FAISS.load_local(
                folder_path=FAISS_INDEX_PATH,
                embeddings=embeddings,
                allow_dangerous_deserialization=True
            )
            print("✅ FAISS向量库加载成功！")
        except Exception as e:
            raise RuntimeError(f"加载FAISS向量库失败：{e}")

        # 构建Agent并启动CLI
        policy_agent = build_agricultural_policy_agent(vector_store)
        interactive_chat(policy_agent, vector_store)