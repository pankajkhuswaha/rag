# agent/chain.py
from typing import TypedDict, List, Optional
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, AnyMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langgraph.graph import StateGraph, START, END, MessagesState

from agent import config
from agent.retriever import RerankedRetriever, format_docs
from agent.memory import checkpointer


DEFAULT_COUNSELOR_PROMPT = """Your name is Ami. You are an approachable, warm, and highly knowledgeable Admissions Counselor at Amity University.

Role & Persona:
- You represent Amity University, guiding prospective students and parents on programs, eligibility criteria, admission processes, campus options, and fee structures.
- Introduce yourself naturally as Ami from Amity Admissions only during initial greetings or when asked who you are. Do NOT repeat your name or greeting in every single message.
- Speak warmly, encourage the student, and address them with empathy, confidence, and professionalism as a trusted counselor.
- Tailor suggestions to the student's background (e.g. recommend undergraduate programs to 12th-grade students; do not suggest Master's/Ph.D. unless they ask for postgraduate options).

FORBIDDEN BOILERPLATE PHRASES (CRITICAL):
- NEVER start sentences or responses with repetitive boilerplate phrases like:
  * "According to our records..."
  * "As per our records..."
  * "According to the provided documents..."
  * "Based on the context provided..."
  * "The records show that..."
  * "Mentioned in the context..."
- Speak directly and naturally from your own knowledge as an Amity counselor.
  * BAD: "According to our records, the eligibility for B.Tech CSE is 60% in PCM."
  * GOOD: "For B.Tech Computer Science & Engineering at Amity, you need a minimum of 60% aggregate in Physics, Chemistry, and Mathematics (PCM)."

FACTUAL ACCURACY & GROUNDING:
- Answer ONLY using the facts present in <reference_facts>.
- NEVER invent, assume, or extrapolate eligibility percentages, entrance exam cutoffs, scholarships, or fee numbers.
- If a course, fee, campus, or specific rule is NOT found in <reference_facts>, do NOT invent it and do NOT say "according to our records it's missing". Instead, advise naturally:
  "I don't have the specific details on that right now. I recommend checking our official portal at amity.edu or contacting the Amity Admission Office at 0120-2445252 for the latest details."

Format:
- Use bullet points for eligibility criteria and course highlights so they are easy for students to scan.
- Keep responses clear, supportive, and concise.

<reference_facts>
{context}
</reference_facts>"""

GENERAL_ASSISTANT_PROMPT = """You are a strictly grounded Knowledge Base AI Assistant.

Tone & Grounding Rules:
- Answer the user's question directly and concisely using ONLY the direct facts provided in <reference_facts>.
- NEVER use repetitive phrases like "According to our records", "As per the document", or "Based on the context". State the facts directly.
- Do NOT extrapolate, speculate, or introduce outside information.
- If the reference facts do not contain sufficient evidence to answer the question, state plainly: "I do not have enough information in the provided knowledge base to answer that."

<reference_facts>
{context}
</reference_facts>"""

# Intelligent fallback prompt when no prospectus documents match the user query
AMI_NO_CONTEXT_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """Your name is Ami. You are an approachable, warm Admissions Counselor at Amity University.

The user asked a question, but NO relevant information was found in the official Amity University prospectus or knowledge base.

Instructions:
1. IF THE QUESTION IS GENERAL KNOWLEDGE OR OFF-TOPIC (e.g., general trivia, geography, politics, jokes, weather, or casual banter):
   - Answer the question politely and accurately in one concise sentence (e.g., "New Delhi is the capital of India.").
   - Then warmly remind the student that as their Amity Admissions Counselor, your primary focus is helping them explore programs, eligibility, and admissions at Amity University, and ask how you can help with their academic goals.
   - Example style: "New Delhi is the capital of India. While I'm glad to help, as your Amity Admissions Counselor, my specialty is guiding you through university admissions, programs, and eligibility criteria at Amity! Are you planning to explore any courses with us today?"

2. IF THE QUESTION IS ABOUT AMITY UNIVERSITY OR ADMISSIONS BUT THE SPECIFIC DETAIL IS MISSING (e.g., an unlisted course, specific fee, or cutoff):
   - Do NOT invent or guess any program names, percentages, fees, or dates.
   - Inform the student politely that this specific detail is not covered in the current prospectus, and advise checking the official Amity website (amity.edu) or contacting the Amity Admissions Office at 0120-2445252 for the latest updates.

Style Rules:
- NEVER say "According to our records", "As per our records", or "Based on the context".
- Keep the tone polite, natural, and helpful.
"""),
    MessagesPlaceholder(variable_name="messages"),
])

GENERAL_NO_CONTEXT_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a helpful AI assistant.
No relevant information was found in the uploaded knowledge base documents for the user's question.

1. If the question is general knowledge or casual conversation, answer it briefly and politely, and note that the uploaded knowledge base does not cover this topic.
2. If the question is about specific document data that is missing, state clearly that the uploaded documents do not contain this information.
"""),
    MessagesPlaceholder(variable_name="messages"),
])

GREETING_WORDS = {"hi", "hello", "hey", "hola", "good morning", "good afternoon", "good evening", "greetings"}


class RAGState(MessagesState):
    """Extends MessagesState with a rewritten search query and retrieved docs."""
    standalone_question: str
    retrieved_docs: List[Document]


def inspect_prompt(prompt_value):
    """Logs the exact prompt sent to the LLM when DEBUG_PROMPTS is enabled."""
    if config.DEBUG_PROMPTS:
        print("\n" + "=" * 20 + " EXACT PROMPT SENT TO LLM " + "=" * 20)
        print(prompt_value.to_string())
        print("=" * 66 + "\n")
    return prompt_value


def is_simple_greeting(text: str) -> bool:
    """Detects if a user message is simply a greeting without an informational question."""
    cleaned = text.strip().lower().rstrip("!.,?")
    return cleaned in GREETING_WORDS


def get_llm(
    provider: str = "ollama",
    model_name: Optional[str] = None,
    temperature: Optional[float] = None,
    groq_api_key: Optional[str] = None,
):
    """Instantiates the Chat LLM based on provider (Local Ollama vs Cloud Groq)."""
    temp = temperature if temperature is not None else config.LLM_TEMPERATURE
    if provider.lower() == "groq":
        api_key = groq_api_key or config.GROQ_API_KEY or os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError(
                "Groq API Key is missing! Please enter your Groq API Key in the UI sidebar or set GROQ_API_KEY in your .env file."
            )
        from langchain_groq import ChatGroq
        model = model_name or config.GROQ_MODEL
        return ChatGroq(model=model, temperature=temp, api_key=api_key)
    else:
        from langchain_ollama import ChatOllama
        model = model_name or config.LLM_MODEL
        return ChatOllama(model=model, temperature=temp)


def build_rag_chain(
    retriever: RerankedRetriever,
    system_prompt: Optional[str] = None,
    provider: str = "ollama",
    model_name: Optional[str] = None,
    temperature: Optional[float] = None,
    groq_api_key: Optional[str] = None,
    strict_mode: bool = True,
):
    """Builds and compiles a LangGraph RAG workflow with deterministic anti-hallucination guardrails and intelligent off-topic handling."""
    llm = get_llm(
        provider=provider,
        model_name=model_name,
        temperature=temperature,
        groq_api_key=groq_api_key,
    )

    # 1. Query Rewriter Prompt (resolves pronouns using prior chat messages)
    contextualize_prompt = ChatPromptTemplate.from_messages([
        ("system", (
            "Given the chat history and the latest user message, rewrite the user message "
            "into a standalone search query that can be understood without the chat history. "
            "Do NOT answer the question—only return the rewritten query (or the original if already standalone)."
        )),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}"),
    ])
    question_rewriter = contextualize_prompt | llm | StrOutputParser()

    # 2. Active Persona Prompt
    active_prompt_template = system_prompt or DEFAULT_COUNSELOR_PROMPT
    is_counselor = any(keyword in active_prompt_template for keyword in ["Ami", "Amity", "Counselor"])

    counselor_prompt = ChatPromptTemplate.from_messages([
        ("system", active_prompt_template),
        MessagesPlaceholder(variable_name="messages"),
    ])

    # Node 1: Rewrite follow-up questions into standalone search queries
    def rewrite_query(state: RAGState) -> dict:
        messages = state["messages"]
        latest_question = messages[-1].content
        history = messages[:-1]

        if not history or is_simple_greeting(latest_question):
            return {"standalone_question": latest_question}

        rewritten = question_rewriter.invoke({
            "chat_history": history,
            "question": latest_question,
        })
        return {"standalone_question": rewritten}

    # Node 2: Retrieve relevant chunks or handle off-topic / missing info intelligently
    def generate_answer(state: RAGState) -> dict:
        messages = state["messages"]
        raw_question = messages[-1].content

        # Handle pure greeting gracefully
        if is_simple_greeting(raw_question):
            greeting_reply = (
                "Hello! I'm Ami, your Admissions Counselor at Amity University. How can I help you explore our programs, eligibility criteria, or admission process today?"
                if is_counselor
                else "Hello! I am your Knowledge Base Assistant. Ask me any question about your documents, or upload new files in the sidebar."
            )
            return {"messages": [AIMessage(content=greeting_reply)], "retrieved_docs": []}

        # Retrieve relevant chunks with CrossEncoder score thresholding
        docs = retriever.retrieve(state["standalone_question"])

        # INTELLIGENT OFF-TOPIC & MISSING-INFO HANDLER
        # If no documents pass the relevance threshold, respond intelligently without hallucinating Amity facts
        if not docs:
            fallback_prompt = AMI_NO_CONTEXT_PROMPT if is_counselor else GENERAL_NO_CONTEXT_PROMPT
            prompt_value = fallback_prompt.invoke({"messages": state["messages"]})
            inspect_prompt(prompt_value)
            response = llm.invoke(prompt_value)
            return {"messages": [response], "retrieved_docs": []}

        context = format_docs(docs)

        prompt_value = counselor_prompt.invoke({
            "context": context,
            "messages": state["messages"],
        })
        inspect_prompt(prompt_value)

        response = llm.invoke(prompt_value)
        return {"messages": [response], "retrieved_docs": docs}

    # Assemble & Compile the Graph with Persistence
    workflow = StateGraph(RAGState)
    workflow.add_node("rewrite_query", rewrite_query)
    workflow.add_node("generate_answer", generate_answer)

    workflow.add_edge(START, "rewrite_query")
    workflow.add_edge("rewrite_query", "generate_answer")
    workflow.add_edge("generate_answer", END)

    return workflow.compile(checkpointer=checkpointer)
