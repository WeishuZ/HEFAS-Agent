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
from langchain.schema import Document

class ChatAgent:
    def __init__(self, knowledge_base_path="data/*", model="gpt-4-mini", db_name="vector_db"):
        # 初始化配置
        self.MODEL = model
        self.db_name = db_name
        self.default_response = ("I am sorry, but I don't have the answer to that. "
                               "However, you can schedule a consultation: "
                               "[Book Appointment](https://daappointments.deanza.edu/hefasonline/eSARS.asp?WCI=Init&WCE=Settings).")
        self.LOG_FILE = "ai_conversation_log.txt"
        self.user_conversations = {}  # 存储不同用户的对话链
        self.system_prompt = """你是 De Anza College HEFAS 项目的AI助手。请严格按照知识库中的内容回答问题。
如果问题超出知识库范围，请直接回复预设的默认回答。请根据用户使用的语言（英文或中文）用相应语言回答。
如果用户使用其他语言提问，请用英文回答。"""
        
        # 加载环境变量
        load_dotenv()
        OPENAI_API_KEY = os.getenv('OPENAI_API_KEY', 'your-key-if-not-using-env')
        print(f"Loaded API Key: {OPENAI_API_KEY}") 
        # 初始化向量数据库
        self.vectorstore = self._initialize_vectorstore(knowledge_base_path)

    def _initialize_vectorstore(self, knowledge_base_path):
        """初始化向量数据库"""
        try:
            # 加载文档
            loader = DirectoryLoader(knowledge_base_path, glob="**/*.txt")
            documents = loader.load()
            
            if not documents:
                print(f"⚠️ 警告：在 {knowledge_base_path} 中没有找到任何文档")
                # 如果没有文档，创建一个包含默认消息的文档
                documents = [Document(page_content="No documents found in knowledge base.")]
            else:
                print(f"✅ 成功加载了 {len(documents)} 个文档")

            # 分割文档
            text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            chunks = text_splitter.split_documents(documents)
            print(f"✅ 文档已分割成 {len(chunks)} 个块")

            # 创建向量数据库
            embeddings = OpenAIEmbeddings()
            vectorstore = Chroma.from_documents(
                documents=chunks, 
                embedding=embeddings,
                persist_directory="./chroma_db"
            )
            print("✅ 向量数据库创建成功")
            
            return vectorstore
            
        except Exception as e:
            print(f"❌ 创建向量数据库时出错：{str(e)}")
            raise

    def _initialize_conversation_chain(self, user_id):
        """为每个用户初始化独立的对话链"""
        llm = ChatOpenAI(
            temperature=0.7, 
            model_name=self.MODEL,
            system_message=self.system_prompt
        )
        memory = ConversationBufferMemory(
            memory_key='chat_history', 
            return_messages=True
        )
        retriever = self.vectorstore.as_retriever(search_kwargs={"k": 25})
        return ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=retriever,
            memory=memory
        )

    def _add_metadata(self, doc, doc_type):
        doc.metadata["doc_type"] = doc_type
        return doc

    def _log_conversation(self, question, history):
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(self.LOG_FILE, "a", encoding="utf-8") as f:
            f.write(f"[{timestamp}]\nUser asked: {question}\nHistory: {history}\n\n")

    def chat(self, question, user_id, history=None):
        """
        处理用户问题并返回回答
        
        Args:
            question (str): 用户的问题
            user_id (str): 用户ID
            history (list, optional): 对话历史记录
            
        Returns:
            str: 回答内容
        """
        # 如果是新用户，为其创建新的对话链
        if user_id not in self.user_conversations:
            self.user_conversations[user_id] = self._initialize_conversation_chain(user_id)
        
        # 使用用户专属的对话链进行对话
        result = self.user_conversations[user_id].invoke({"question": question})
        
        # 如果AI表示不知道答案，返回默认回答
        if "I don't know" in result["answer"].lower() or "不知道" in result["answer"]:
            result["answer"] = self.default_response
            self._log_conversation(question, history)
            
        return result["answer"]