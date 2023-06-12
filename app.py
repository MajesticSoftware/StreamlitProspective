from langchain.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain import OpenAI
import os
from gtts import gTTS
import streamlit as st


embeddings = OpenAIEmbeddings(openai_api_key=os.environ["OPENAI_API_KEY"])

loader = DirectoryLoader('ScrapData', glob="**/*.txt")
documents = loader.load()

text_splitter = CharacterTextSplitter(chunk_size=2500, chunk_overlap=0)
texts = text_splitter.split_documents(documents)

docsearch = Chroma.from_documents(texts, embeddings)
qa = RetrievalQA.from_chain_type(
    llm=OpenAI(),
    chain_type="stuff",
    retriever=docsearch.as_retriever()
)
count = 0
def query(q):
    print("Query: ", q)
    answer = qa.run(q)
    print("Answer: ", answer)
    return answer

def soundQuery(q):
    global count
    print("Query: ", q)
    answer = qa.run(q)
    tts = gTTS(answer)
    count += 1
    tts.save(f'answer{count}.mp3')
    print("Answer: ", answer)

st.title("LeckyAI")

if "history" not in st.session_state:
    st.session_state["history"] = []

user_question = st.text_input("Enter your question:")
if st.button("Submit"):
    response = query(user_question)
    #soundQuery(user_question)

    st.session_state["history"].append((user_question, response))

    for idx, (question, answer) in enumerate(st.session_state["history"], 1):
        st.write(f"{idx}. Question: {question}")
        st.write(f"Answer: {answer}")
        #st.audio(f'answer{count}.mp3')
