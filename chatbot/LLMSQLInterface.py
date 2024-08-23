import os
import json
from google.cloud import bigquery
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import create_sql_agent
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.example_selectors import SemanticSimilarityExampleSelector
from langchain_openai import OpenAIEmbeddings
from langchain_core.prompts import (
    ChatPromptTemplate,
    FewShotPromptTemplate,
    MessagesPlaceholder,
    PromptTemplate,
    SystemMessagePromptTemplate,
)
from sqlalchemy import *
from sqlalchemy.schema import *
import datetime
from bigq_insert import BQDataInserter 
import pandas as pd

with open('.config.json') as f:
    config_data = json.load(f)

os.environ["OPENAI_API_KEY"] = config_data["OPENAI_API_KEY"]

def get_sqlalchemy_url():
    gcp_project = "new-chatbot-428815"
    client = bigquery.Client(project=gcp_project)
    dataset = "Traffic"
    return f'bigquery://{gcp_project}/{dataset}?'    
    
class LLMSQLInterface:
    def __init__(self):
        sqlalchemy_url = get_sqlalchemy_url()
        self.db = SQLDatabase.from_uri(sqlalchemy_url)
        self.llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
        self.toolkit = SQLDatabaseToolkit(db=self.db, llm=self.llm)
        
        self.bq_inserter = BQDataInserter()
        
        self.df_insert = None
        
        self.examples = [
            {"input": "How many road incidents are there today?", 
            "query": "SELECT COUNT(*) AS total_incident FROM `new-chatbot-428815.Traffic.traffic_incidents` WHERE Message LIKE CONCAT('(', DATE_FORMAT(NOW(), '%d/%c'), '%');"
            },
            {"input": "How many traffic incidents are there today?", 
            "query": "SELECT COUNT(*) AS total_incident FROM `new-chatbot-428815.Traffic.traffic_incidents` WHERE Message LIKE CONCAT('(', DATE_FORMAT(NOW(), '%d/%c'), '%');"
            },
            {"input": "How many road incidents are there yesterday?", 
            "query": "SELECT COUNT(*) AS total_incident FROM `new-chatbot-428815.Traffic.traffic_incidents` WHERE Message LIKE CONCAT('(', DATE_FORMAT(NOW() - INTERVAL 1 DAY, '%d/%c'), '%');"
            },
            {"input": "How many traffic incidents are there yesterday?", 
            "query": "SELECT COUNT(*) AS total_incident FROM `new-chatbot-428815.Traffic.traffic_incidents` WHERE Message LIKE CONCAT('(', DATE_FORMAT(NOW() - INTERVAL 1 DAY, '%d/%c'), '%');"
            }, 
            {"input": "How many road works are there today?",
            "query": 'SELECT COUNT(*) AS total_roadworks FROM `new-chatbot-428815.Traffic.road_works` WHERE StartDate <= CURDATE() AND EndDate >= CURDATE();'
            },
            {"input": "Which area experienced the most number of road works?",
            "query": "SELECT RoadName AS Area, COUNT(*) AS total_roadworks FROM `new-chatbot-428815.Traffic.road_works` GROUP BY RoadName ORDER BY total_roadworks DESC LIMIT 1"
            },
            {"input": "What is the traffic condition now?",
            "query": "SELECT RoadName, Volume, HourOfDate FROM `new-chatbot-428815.Traffic.traffic_flow` ORDER BY Date, HourOfDate DESC LIMIT 10;"
            },
            {"input": "How many car parks are available in the star vista right now?",
            "query": 'SELECT AvailableLots FROM `new-chatbot-428815.Traffic.carpark_avail` WHERE Development = "The Star Vista";'
            } ,
            {"input": "Where can I find the most car park in Orchard area?",
            "query": 'SELECT MAX(AvailableLots) As MaxLots FROM `new-chatbot-428815.Traffic.carpark_avail` WHERE Area = "Orchard";'
            }, 
            {"input": "How is the traffic flow today?",
             "query": "SELECT RoadName, Volume FROM `new-chatbot-428815.Traffic.traffic_flow` WHERE Date = CURDATE() ORDER BY HourOfDate DESC LIMIT 10"
            },
            {"input": "What is the estimated travelling time from Orchard Road to Havelock?",
             "query": "SELECT StartPoint, EndPoint, EstTime FROM `new-chatbot-428815.Traffic.estimated_travel_times` WHERE StartPoint LIKE '%Orchard%' AND EndPoint LIKE '%Havelock%';"
            },
            {"input": "How many bus stops are there along Victoria Street?",
             "query": "SELECT COUNT(DISTINCT(BusStopCode)) AS TotalBusStop FROM `new-chatbot-428815.Traffic.bus_stops` WHERE RoadName = 'Victoria St';"
            },
            {"input": "Which road has the least number of bus stops?",
             "query": "SELECT RoadName, COUNT(DISTINCT(BusStopCode)) AS Total_Bus_Stop FROM `new-chatbot-428815.Traffic.bus_stops` GROUP BY RoadName ORDER BY Total_Bus_Stop, RoadName ASC LIMIT 5;"
            },
            {"input": "Which road has the most number of bus stops?",
             "query": "SELECT RoadName, COUNT(DISTINCT(BusStopCode)) AS Total_Bus_Stop FROM `new-chatbot-428815.Traffic.bus_stops` GROUP BY RoadName ORDER BY Total_Bus_Stop, RoadName DESC LIMIT 5;"
            },
            {"input": "Given the public transport utilization target of 75% by 2030, can we reach the target?",
             "query": "SELECT * FROM `new-chatbot-428815.Chatbot.public_transport_utilization WHERE year=2030`;"
            },
            {"input": "Given the emission target for transport of 6,000,000,000 kg CO2 by 2030, can we reach the target?",
             "query": "SELECT * FROM `new-chatbot-428815.Chatbot.carbon_emission WHERE year=2030`;"
            }
        ]

        self.system_prefix = """You are an agent designed to interact with a SQL database.
            Given an input question, create a syntactically correct {dialect} query to run, then look at the results of the query and return the answer.
            Unless the user specifies a specific number of examples they wish to obtain, always limit your query to at most {top_k} results.
            You can order the results by a relevant column to return the most interesting examples in the database.
            Never query for all the columns from a specific table, only ask for the relevant columns given the question.
            You have access to tools for interacting with the database.
            Only use the given tools. Only use the information returned by the tools to construct your final answer.
            You MUST double check your query before executing it. If you get an error while executing a query, rewrite the query and try again.

            DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.) to the database.

            Here are some examples of user inputs and their corresponding SQL queries:"""
        
        self.example_selector = SemanticSimilarityExampleSelector.from_examples(
            self.examples,
            OpenAIEmbeddings(),
            FAISS,
            k=5,
            input_keys=["input"],
        )

        self.few_shot_prompt = FewShotPromptTemplate(
            example_selector=self.example_selector,
            example_prompt=PromptTemplate.from_template(
                "User input: {input}\nSQL query: {query}"
            ),
            input_variables=["input", "dialect", "top_k"],
            prefix=self.system_prefix,
            suffix="",
        )

        self.full_prompt = ChatPromptTemplate.from_messages(
            [
                SystemMessagePromptTemplate(prompt=self.few_shot_prompt),
                ("human", "{input}"),
                MessagesPlaceholder("agent_scratchpad"),
            ]
        )

        self.agent = create_sql_agent(
            llm=self.llm,
            toolkit=self.toolkit,
            prompt=self.full_prompt,
            verbose=True,
            agent_type="openai-tools",
        )

    def create_conversation(self, query: str, chat_history: list) -> tuple:
        result = self.agent.invoke({"input": query})['output']
        chat_history.append((query, result))

        cur_timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        self.bq_inserter.create_table()
            
        self.df_insert = pd.DataFrame({
            'file_name': ['Database Query'], 
            'query': [query], 
            'answer': [result], 
            'timestamp': [cur_timestamp]
        })
        
        self.bq_inserter.insert_dataframe(self.df_insert)
        
        return '', chat_history

            