# services/llm_service.py
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_message_histories import ChatMessageHistory
from config.settings import Settings

class LLMService:
    def __init__(self):
        self.llm = ChatOpenAI(
            openai_api_key=Settings.OPENROUTER_API_KEY,
            openai_api_base="https://openrouter.ai/api/v1",
            model_name=Settings.llm_model,
            temperature=0.7
        )
        # هنا بنعرف مخزن الذاكرة في الرام (دي بتتمسح لو قفلت البرنامج)
        self.history = ChatMessageHistory()

    def get_answer(self, query, context):
        """
        query: سؤال الطالب
        context: المعلومات اللي رجعت من الـ Vector Store
        """
        # trimmed_history = self.history.messages[-4:] if len(self.history.messages) > 4 else self.history.messages
        
        # الـ Prompt السحري باللهجة المصرية
        template = """
        You are "Study Companion," an intelligent and attentive study assistant. Your task is to help the student or researcher fully understand the material based only on the information provided in the context.

Context to use:
{context}

Student/Researcher Question:
{query}

        Answering Rules:
        1. Always respond in the same language, tone, and style as the query.
        2. **Strictly Source-Based**: Use ONLY the information provided in the context for facts. Do NOT use outside knowledge for facts.
        3. **Flexible Explanation**: You MAY explain the provided facts in new ways, give original analogies, or rephrase for clarity, as long as the underlying information comes from the context.
        4. **No Tables**: Do NOT generate tables. Use bullet points or lists if structure is needed.
        5. If the answer is not present in the context, respond politely:
           "This information is not available in the uploaded file."
        6. Explain the answer in **multiple layers**:
           - **Beginner-friendly:** simple and clear explanation with an analogy.
           - **Advanced:** deeper explanation.
           - **Detailed/Specific:** all relevant details.
        7. Provide a **summary** at the end.
        
        Your mission: Deliver a context-based, fully accurate explanation that mirrors the user's style. No tables. Use analogies and rephrasing to make the context clear.
        """

        prompt = ChatPromptTemplate.from_messages([
            ("system", template),
            MessagesPlaceholder(variable_name="chat_history"), # هنا التاريخ هيتحط
            ("user", "{query}")
        ])
        
        chain = prompt | self.llm | StrOutputParser()

        # تشغيل الـ Chain مع تمرير التاريخ الحالي
        response = chain.invoke({
            "query": query,
            "context": context,
            "chat_history": self.history.messages
        })

        # أهم خطوة: بنسيف السؤال والرد في الـ History عشان المرة الجاية
        self.history.add_user_message(query)
        self.history.add_ai_message(response)

        return response


    def rewrite_query(self, query):
        """
        Hyper-detailed Query Rewriting:
        Transforms the user query into a descriptive, context-rich 'search prompt' 
        to optimize vector retrieval accuracy.
        """
        if not self.history.messages:
            # If no history, we still want to expand the single query to be more descriptive
            chat_history = "No previous history."
        else:
            chat_history = self.history.messages
            
        rewrite_template = """
You are an expert Query Optimizer for a RAG (Retrieval-Augmented Generation) system.
Your goal is to transform the user's question into a **highly detailed, descriptive search query** that will maximize the chances of finding the relevant technical passages in a dense vector database.

- **Conversation History**: {chat_history}
- **Student's Current Question**: {query}

### INSTRUCTIONS:
1. **Language & Style Matching**: The optimized prompt **MUST** be in the same language, tone, and delicate style as the student's question.
2. **Context Expansion**: Replace all pronouns ("it", "this", "those") with specific entities from the history.
3. **Predictive Detailing (Query Expansion)**: 
   - Expand the query into a detailed descriptive paragraph.
   - Include likely technical terms, specific concepts, and related terminology.
   - **Important**: Keep technical terms in English (or their most globally recognized form) if they are more precise or common that way in the original language's context.
4. **Retrieval Focus**: Formulate the output as a statement that describes the information being sought, as it would appear in a professional textbook or research paper.
5. **Constraint**: Output **ONLY** the optimized search prompt. Do not add any conversational filler.
        """
        prompt = ChatPromptTemplate.from_template(rewrite_template)
        chain = prompt | self.llm | StrOutputParser()
        return chain.invoke({"chat_history": chat_history, "query": query})
