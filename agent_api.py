import os
import glob
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
import datetime

class ChatAgent:
    def __init__(self, knowledge_base_path="HEFAS_Knowledge/*", model="gpt-4o-mini", db_name="vector_db"):
        # 初始化配置
        self.MODEL = model
        self.db_name = db_name
        self.default_response = ("I am sorry, but I don't have the answer to that. "
                               "However, you can schedule a consultation: "
                               "[Book Appointment](https://daappointments.deanza.edu/hefasonline/eSARS.asp?WCI=Init&WCE=Settings).")
        self.LOG_FILE = "ai_conversation_log.txt"
        
        # 加载环境变量
        load_dotenv()
        OPENAI_API_KEY = os.getenv('OPENAI_API_KEY', 'your-key-if-not-using-env')
        print(f"Loaded API Key: {OPENAI_API_KEY}") 
        # 初始化向量数据库
        self.vectorstore = self._initialize_vectorstore(knowledge_base_path)
        
        # 初始化对话链
        self._initialize_conversation_chain()

    def _initialize_vectorstore(self, knowledge_base_path):
        # 读取文档
        folders = glob.glob(knowledge_base_path)
        documents = []
        text_loader_kwargs = {'encoding': 'utf-8'}
        
        for folder in folders:
            doc_type = os.path.basename(folder)
            loader = DirectoryLoader(folder, glob="**/*.md", loader_cls=TextLoader, 
                                   loader_kwargs=text_loader_kwargs)
            folder_docs = loader.load()
            documents.extend([self._add_metadata(doc, doc_type) for doc in folder_docs])

        # 分割文档
        text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = text_splitter.split_documents(documents)
        
        # 创建向量存储
        embeddings = OpenAIEmbeddings()
        if os.path.exists(self.db_name):
            Chroma(persist_directory=self.db_name, 
                  embedding_function=embeddings).delete_collection()
            
        return Chroma.from_documents(documents=chunks, embedding=embeddings, 
                                   persist_directory=self.db_name)

    def _initialize_conversation_chain(self):
        llm = ChatOpenAI(temperature=0.7, model_name=self.MODEL)
        self.memory = ConversationBufferMemory(memory_key='chat_history', 
                                             return_messages=True)
        retriever = self.vectorstore.as_retriever(search_kwargs={"k": 25})
        self.conversation_chain = ConversationalRetrievalChain.from_llm(
            llm=llm, 
            retriever=retriever, 
            memory=self.memory
        )

    def _add_metadata(self, doc, doc_type):
        doc.metadata["doc_type"] = doc_type
        return doc

    def _log_conversation(self, question, history):
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(self.LOG_FILE, "a", encoding="utf-8") as f:
            f.write(f"[{timestamp}]\nUser asked: {question}\nHistory: {history}\n\n")

    def chat(self, question, history=None):
        """
        处理用户问题并返回回答
        
        Args:
            question (str): 用户的问题
            history (list, optional): 对话历史记录
            
        Returns:
            str: 回答内容
        """
        result = self.conversation_chain.invoke({"question": question})
        if result["answer"] == "I don't know.":
            result["answer"] = self.default_response
            self._log_conversation(question, history)
        return result["answer"] 