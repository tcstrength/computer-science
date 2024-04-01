from langchain_community.chat_models import ChatCohere
from langchain_core.prompts import ChatPromptTemplate

class InferenceService():
    def __init__(self, vector_search):
        llm = ChatCohere()
        prompt = ChatPromptTemplate.from_messages([
            ("ai", open("rag/inference_template.txt").read())
        ])
        self.retriever = vector_search
        self.chain = prompt | llm
        
    def invoke(self, question):
        context = self.retriever.similarity_search(question, k=1)[0].page_content
        return self.chain.invoke({
            "context": context, "question": question
        })