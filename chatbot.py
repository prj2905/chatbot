import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader, WebBaseLoader

from langchain.chains import ConversationalRetrievalChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.memory import ConversationBufferMemory

load_dotenv()

# web_urls = [
#     ""
# ]

# web_loader = WebBaseLoader(web_urls)
# web_docs = web_loader.load()

file_loader = TextLoader("bot_content.md",encoding="utf-8")
file_docs = file_loader.load()

docs =  file_docs

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
chunks = splitter.split_documents(docs)


embedding = GoogleGenerativeAIEmbeddings(model="models/embedding-001")



if os.path.exists("faiss_index"):
    vectorstore = FAISS.load_local("faiss_index", embedding, allow_dangerous_deserialization=True)
else:
    vectorstore = FAISS.from_documents(chunks, embedding)
    vectorstore.save_local("faiss_index")

llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    google_api_key=os.getenv("GOOGLE_API_KEY")
)


memory = ConversationBufferMemory(
    memory_key="chat_history", return_messages=True
)


qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=vectorstore.as_retriever(),
    memory=memory,
    verbose=True
)



def get_bot_response(query: str) -> str:
    return qa_chain.run({"question": query})
