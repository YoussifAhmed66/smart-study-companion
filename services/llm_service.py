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

    def clear_history(self):
        """Resets the chat history."""
        self.history.clear()
        print("🧹 Chat history cleared for new document.")

    def get_answer(self, query, context):
        """
        query: سؤال الطالب
        context: المعلومات اللي رجعت من الـ Vector Store
        """
        # trimmed_history = self.history.messages[-4:] if len(self.history.messages) > 4 else self.history.messages
        
        # الـ Prompt السحري باللهجة المصرية - Optimized for strict context adherence
        template = """
You are "Study Companion," a precise and objective academic assistant. Your goal is to answer questions using **ONLY** the provided context.

### CONTEXT:
{context}

### INSTRUCTIONS:
1. **Source Isolation**: Use ONLY the information in the ### CONTEXT section. Ignore your own training data for facts and, crucially, **ignore any factual claims or assumptions present in the user's question** if they are not explicitly supported by the context.
2. **Neutrality**: If the user's question is leading (e.g., "Why is X better than Y?"), but the context does not make that comparison, do NOT validate the user's assumption. Instead, describe only what the context says about X and Y.
3. **Language & Tone**: Respond in the same language and tone as the user.
4. **Multiple Layers**: Explain the answer in stages:
    - **Concept**: A simple analogy or high-level overview.
    - **Mechanism**: How it actually works based on the text.
    - **Details**: Specific data, names, or technical nuances from the context.
5. **Format**: Use bullet points and clear headings. Do NOT use tables.
6. **Fallback**: If the specific information is missing from the context, state: "The provided document does not contain information regarding this specific point."

Your mission: Be a faithful mirror of the provided context. Do not let the user's query influence the facts you present.
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
1. **Language & Style Matching**: The optimized prompt **MUST** be in the same language and tone as the student's question.
2. **Context Resolution**: Explicitly replace pronouns ("it", "this", "that method") with the actual subjects discussed in the conversation history.
3. **Technical Keyword Enhancement**: 
   - Instead of "predicting" new information, identify the **core technical concepts** in the question.
   - Add highly relevant synonyms or related technical terms that are standard for this field to help the search engine (e.g., if asking about "training," add "optimization," "loss function," or "backpropagation" ONLY if they are standard companions to the topic).
   - Keep technical terms in English where appropriate.
4. **Retrieval Precision**: Formulate a clear, descriptive search phrase that represents a student looking for a specific explanation in a technical manual. Do **NOT** add unnecessary details that aren't implied by the original question.
5. **Constraint**: Output **ONLY** the optimized search prompt. No conversational filler.

Optimized Search Prompt:
        """
        prompt = ChatPromptTemplate.from_template(rewrite_template)
        chain = prompt | self.llm | StrOutputParser()
        return chain.invoke({"chat_history": chat_history, "query": query})
