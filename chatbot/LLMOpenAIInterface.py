from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain.document_loaders.csv_loader import CSVLoader
from langchain_community.document_loaders import Docx2txtLoader
from langchain_community.document_loaders import UnstructuredExcelLoader
from langchain_openai import OpenAI
from langchain_openai import OpenAIEmbeddings
import pandas as pd
import json
import os
import shutil
import datetime
from bigq_insert import BQDataInserter 

class docRetriever:
    def __init__(self, embed_model_name, model_kwargs, chunk_size=1000, chunk_overlap=100):
        self.embeddings = OpenAIEmbeddings(model=embed_model_name)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.document_path = None
        self.db = None
        self.db_intermediate = None
        self.list_of_documents = None
    
    def split_doc(self, document_path, verbose):
        self.document_path = document_path
        if document_path.endswith("csv"):
            if verbose: print("csv loaded")
            loader = CSVLoader(document_path) 
        elif document_path.endswith("pdf"):
            if verbose:print("pdf loaded")
            loader = PyPDFLoader(document_path)
        elif document_path.endswith("txt"):
            if verbose:print("txt loaded")
            loader = TextLoader(document_path)
        elif document_path.endswith("docx"):
            if verbose:print("docx loaded")
            loader = Docx2txtLoader(document_path)
        elif document_path.endswith("xlsx"):
            if verbose:print("xlsx loaded")
            loader = UnstructuredExcelLoader(document_path)          
        else:
            if verbose:print("No document found.")
                
        raw_documents = loader.load()
        text_splitter = CharacterTextSplitter(chunk_size=self.chunk_size, chunk_overlap=self.chunk_size)
        documents = text_splitter.split_documents(raw_documents)
        
        if not self.db_intermediate:
            self.db_intermediate = FAISS.from_documents(documents, self.embeddings)
        else:
            self.db_add = FAISS.from_documents(documents, self.embeddings)
            self.db_intermediate.merge_from(self.db_add)
        
        self.list_of_documents = list(self.db_intermediate.docstore._dict.values())
        self.db = FAISS.from_documents(self.list_of_documents, self.embeddings)

class LLMOpenAI:
    def __init__(self):
        self.llm = OpenAI(temperature=0)
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
        self.chat_histories = {'file_name': [], 'query': [], 'answer': [], 'timestamp': []}
        self.output_file_paths = []
        self.files = []
        self.output_file_path = None
        self.doc_retriever = None
        self.memory = None
        self.file_name = None
        self.bq_inserter = BQDataInserter()
        self.df_insert = None
        self.db = None
        self.selected_doc = None
        
    def create_conversation(self, query: str, chat_history: list) -> tuple:
        try:
            self.selected_doc = self.db.similarity_search(query, filter=dict(source=self.output_file_paths[-1]), k=5, fetch_k=15)
            
            if len(self.selected_doc) == 0:
                self.selected_doc = [self.db.similarity_search_with_score(query + self.output_file_paths[-1])[0][0]]
            
            faiss_to_doc = FAISS.from_documents(documents=self.selected_doc, embedding=self.embeddings)
            retriever = faiss_to_doc.as_retriever()
            
            qa_chain = ConversationalRetrievalChain.from_llm(
                llm=self.llm,
                retriever=retriever,
                memory=self.memory,
                get_chat_history=lambda h: h,
            )
            result = qa_chain({'question': query, 'chat_history': chat_history})
            chat_history.append((query, result['answer']))

            cur_timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            self.chat_histories['file_name'].append(self.file_name)
            self.chat_histories['query'].append(query)
            self.chat_histories['answer'].append(result['answer'])
            self.chat_histories['timestamp'].append(cur_timestamp)

            self.bq_inserter.create_table()
            
            self.df_insert = pd.DataFrame({
                'file_name': [self.file_name], 
                'query': [query], 
                'answer': [result['answer']], 
                'timestamp': [cur_timestamp]
            })
            
            self.bq_inserter.insert_dataframe(self.df_insert)
            
            return '', chat_history

        except Exception as e:
            chat_history.append((query, e))
            return '', chat_history

    def process_file(self, File):
        directory = './flagged'
        try:
            shutil.rmtree(directory)
            print("Deleted directory:", directory)
        except Exception as e:
            print("Failed to delete directory:", directory, e)
            
        self.doc_retriever = docRetriever(embed_model_name="text-embedding-3-large", model_kwargs={"device": "cpu"})
        self.memory = ConversationBufferMemory(memory_key='chat_history', return_messages=False)
        self.file_name = os.path.basename(File)
        output_file_path = os.path.join(os.getcwd(), "uploaded_files", self.file_name)
        shutil.copyfile(File.name, output_file_path)

        if output_file_path not in self.output_file_paths:
            self.files.append(self.file_name)
            self.output_file_paths.append(output_file_path)
            msg = f"File: < {self.file_name} > uploaded.\n\nLoaded files:\n{self.files}"
        else:
            self.files.remove(self.file_name)
            self.output_file_paths.remove(output_file_path)
            self.files.append(self.file_name)
            self.output_file_paths.append(output_file_path)
            msg = f"File: < {self.file_name} > already uploaded.\n\nLoaded files:\n{self.files}"

        self.doc_retriever.split_doc(output_file_path, verbose=False)
        self.db = self.doc_retriever.db

        return msg

if __name__ == "__main__":
    with open('.config.json') as f:
        config_data = json.load(f)

    os.environ["OPENAI_API_KEY"] = config_data["OPENAI_API_KEY"]

