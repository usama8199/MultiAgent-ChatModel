#!/usr/bin/env python

import getpass
import os
from langchain_core.messages import HumanMessage
from langchain_openai import AzureChatOpenAI
from langchain_openai import AzureOpenAIEmbeddings
import pickle
import dill
from dotenv import load_dotenv

load_dotenv()

TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY")
AZURE_OPENAI_API_KEY = os.environ.get("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.environ.get("AZURE_OPENAI_ENDPOINT")
LANGCHAIN_API_KEY = os.environ.get("LANGCHAIN_API_KEY")
SLACK_BOT_TOKEN = os.environ.get("SLACK_BOT_TOKEN")

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "multiagent_rag"


llm = AzureChatOpenAI(
    openai_api_version="api-version",
    azure_deployment="deployement_name",streaming=True
)

Embeddings_model = AzureOpenAIEmbeddings(
    openai_api_version="api-version",
    azure_deployment="deployement_name"
)


from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

def load_retriever(directory='retriever_storage'):
    retriever_path = os.path.join(directory, 'retriever.pkl')
    with open(retriever_path, 'rb') as f:
        retriever = dill.load(f)
    
    vectorstore = Chroma.from_documents(
    documents=retriever,
    collection_name="pdf-chroma",
    embedding=Embeddings_model,
    )

    retriever = vectorstore.as_retriever(search_type="mmr",search_kwargs={"k": 5})
    return retriever


from typing import Dict, TypedDict

from langchain_core.messages import BaseMessage


class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        keys: A dictionary where each key is a string.
    """

    keys: Dict[str, any]


import json
import operator
from typing import Annotated, Sequence, TypedDict

from langchain import hub
from langchain.output_parsers.openai_tools import PydanticToolsParser
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.vectorstores import Chroma
from langchain_core.messages import BaseMessage, FunctionMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.runnables import RunnablePassthrough
from langchain_core.utils.function_calling import convert_to_openai_tool
from langchain_openai import ChatOpenAI, OpenAIEmbeddings



def retrieve(state):

    """
    Retrieve documents

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, documents, that contains retrieved documents
    """
    print("---RETRIEVE---")
    state_dict = state["keys"]
    question = state_dict["question"]
    
    retriever = load_retriever()
    documents = retriever.get_relevant_documents(question)
    old_question=state_dict["old_question"]
    return {"keys": {"documents": documents, "question": question,"old_question":old_question}}


def generate(state):
    """
    Generate answer

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, generation, that contains LLM generation
    """
    print("---GENERATE---")
    state_dict = state["keys"]
    question = state_dict["question"]
    documents = state_dict["documents"]
    old_question= state_dict["old_question"]
    # Prompt
    prompt = hub.pull("rlm/rag-prompt")

    # LLM
    #llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0, streaming=True)

    # Post-processing
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # Chain
    rag_chain = prompt | llm | StrOutputParser()

    # Run
    generation = rag_chain.invoke({"context": documents, "question": question})
    return {
        "keys": {"documents": documents, "question": old_question, "generation": generation}
    }


def grade_documents(state):
    """
    Determines whether the retrieved documents are relevant to the question.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Updates documents key with relevant documents
    """

    print("---CHECK RELEVANCE---")
    state_dict = state["keys"]
    question = state_dict["question"]
    documents = state_dict["documents"]
    old_question=state_dict["old_question"]
    # Data model
    class grade(BaseModel):
        """Binary score for relevance check."""

        binary_score: str = Field(description="Relevance score 'yes' or 'no'")

    # LLM
    model = llm

    # Tool
    grade_tool_oai = convert_to_openai_tool(grade)

    # LLM with tool and enforce invocation
    llm_with_tool = model.bind(
        tools=[grade_tool_oai],
        tool_choice={"type": "function", "function": {"name": "grade"}},
    )

    # Parser
    parser_tool = PydanticToolsParser(tools=[grade])

    # Prompt
    prompt = PromptTemplate(
        template="""You are a grader assessing relevance of a retrieved document to a user question. \n 
        Here is the retrieved document: \n\n {context} \n\n
        Here is the user question and/or generated output from previous agent: {question} \n
        If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. \n
        Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question.""",
        input_variables=["context", "question"],
    )

    # Chain
    chain = prompt | llm_with_tool | parser_tool

    # Score
    filtered_docs = []
    search = "Yes"  # Default do not opt for web search to supplement retrieval
    for d in documents:
        score = chain.invoke({"question": question, "context": d.page_content})
        grade = score[0].binary_score
        if grade == "yes":
            print("---GRADE: DOCUMENT RELEVANT---")
            filtered_docs.append(d)
            search = "No"
        else:
            print("---GRADE: DOCUMENT NOT RELEVANT---")
            # Perform web search
            continue
    print("----grading done---")
    return {
        "keys": {
            "documents": filtered_docs,
            "question": question,
            "run_web_search": search,
            "old_question":old_question
        }
    }


def transform_query_web(state):
    """
    Transform the query to produce a better question.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Updates question key with a re-phrased question
    """

    print("---TRANSFORM QUERY WEB---")
    state_dict = state["keys"]
    question = state_dict["question"]
    documents = state_dict["documents"]
    old_question=state_dict['old_question']
    print("Web Question"+question)
    # Create a prompt template with format instructions and the query
    prompt = PromptTemplate(
        template="""You are generating questions that is well optimized for retrieval from web. \n 
        Look at the input and try to reason about the underlying sematic intent / meaning. \n 
        Here is the initial question:
        \n ------- \n
        {question} 
        \n ------- \n
        Formulate an improved question: """,
        input_variables=["question"],
    )

    # Grader
    model = llm

    # Prompt
    chain = prompt | model | StrOutputParser()
    better_question = chain.invoke({"question": question})
    print("Better Question"+better_question)
    return {"keys": {"documents": documents, "question": better_question,"old_question":old_question}}

def transform_query_retriver(state):
    """
    Transform the query to produce a better question.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Updates question key with a re-phrased question
    """

    print("---TRANSFORM QUERY RETRIVER---")
    state_dict = state["keys"]
    question = state_dict["question"]
    print("Retriver Question: "+question)
    # Create a prompt template with format instructions and the query
    prompt = PromptTemplate(
        template="""Your task is to generate questions that are optimized for retrieving information from a document. 
                Analyze the input question and/or generation and try to understand the underlying semantic intent or meaning. 
                If the generation is present that means the generated output was not satisfctory.
                Below is the initial question and/or the output generated by a previous agent (which may not have provided a correct or relevant answer on its first attempt in that case you can also decide rewrite the question relevent to web search):
                \n ------- \n
                {question} 
                \n ------- \n
                Based on this, formulate an improved question:""",
        input_variables=["question"],
    )

    # Grader
    model = llm

    # Prompt
    chain = prompt | model | StrOutputParser()
    better_question = chain.invoke({"question": question})
    print("Better Question "+better_question)
    return {"keys": {"question": better_question,"old_question":question}}


def web_search(state):
    """
    Web search based on the re-phrased question using Tavily API.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Updates documents key with appended web results
    """

    print("---WEB SEARCH---")
    state_dict = state["keys"]
    question = state_dict["question"]
    documents = state_dict["documents"]
    old_question=state_dict["old_question"]
    tool = TavilySearchResults()
    docs = tool.invoke({"query": question})
    web_results = "\n".join([d["content"] for d in docs])
    web_results = Document(page_content=web_results)
    documents.append(web_results)

    return {"keys": {"documents": documents, "question": question,"old_question":old_question}}


### Edges


def decide_to_generate(state):
    """
    Determines whether to generate an answer or re-generate a question for web search.

    Args:
        state (dict): The current state of the agent, including all keys.

    Returns:
        str: Next node to call
    """

    print("---DECIDE TO GENERATE---")
    state_dict = state["keys"]
    question = state_dict["question"]
    filtered_documents = state_dict["documents"]
    search = state_dict["run_web_search"]
    old_question=state_dict["old_question"]
    if search == "Yes":
        # All documents have been filtered check_relevance
        # We will re-generate a new query
        print("---DECISION: TRANSFORM QUERY and RUN WEB SEARCH---")
        return "transform_query_web"
    else:
        # We have relevant documents, so generate answer
        print("---DECISION: GENERATE---")
        return "generate"




from langchain.output_parsers.openai_functions import JsonOutputFunctionsParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import LLMChain
from langchain_community.agent_toolkits import GmailToolkit
from langchain_community.tools.gmail.utils import (
    build_resource_service,
    get_gmail_credentials,
)
from langchain import hub
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain_openai import ChatOpenAI
from langchain_community.agent_toolkits import SlackToolkit
from langchain.agents import Tool, AgentExecutor, initialize_agent, create_react_agent
from langchain.agents import AgentType, initialize_agent

def supervisor_main(state):
    print("---SUPERVISOR MAIN---")
    members = ["transform_query_retriver","GPTGenericInformation","GmailReadWriteAgent","SlackAgent"]
    system_prompt = (
    """
    As a supervisor, your role is to manage a conversation involving the following team members: {members}. When a user request is received, your responsibility is to determine which team member should respond next. Each team member has a unique skill set and will perform a task based on their expertise, providing a generated answer. I will supply you with the user's Question and/or the current Generation to assist you in deciding which team members to engage.
    For specific questions or non-retrieval questions, for example:
    - Calculations: "Can you convert 15 degrees Celsius to Fahrenheit?", "How do you calculate the compound interest on a principal amount of $5000, with an annual interest rate of 5%, compounded quarterly, for 2 years?"
    - General knowledge: "What is the capital of France?", "Who wrote the book '1984'?", "What are the implications of quantum computing for data security?"
    - Coding: "Can you write a Python function to calculate the factorial of a number?"
    Retrieval(transform_query_retriver) examples could be:
    - "What are the three main points made in the conclusion of the research paper?"
    - "What is the main argument of the third chapter in the book?"
    - "What are the steps outlined in the tutorial to install the software?"
    - "What are the side effects listed for the medication in the product leaflet?"
    - "What is the company's revenue for the 2020 fiscal year as reported in the annual report?"
    Mail (Gmail) related examples could be:
    - "Write a mail asking for leave permission"
    - "What will be my latest mail?"
    Note: After asking mail to do certain task, there is a chance it will ask user consent with something like "I have drafted the email for you. could you please confirm the content to be sent?" which is more question or waiting for user based This also means your work has been completed and you can respond with "FINISH" as user have to act next.
    Slack related examples could be:
    -"Send greeting to coworkers in #general channel"
    -"Get the chat history from the #general channel"
    Note: After asking slack to do certain task, there is a chance it will ask user consent with something like "I have created the draft on the time and place of the meet, could you please confirm to sent the draft?" which is more question or waiting for user based This also means your work has been completed and you can respond with "FINISH" as user have to act next.
    Note: The above examples are just examples for you to think and not the ultimate question which will be asked
    
    If you have any doubts on which specific tool to use, prefer the Retriever. You can use multiple tools if required. Use the question as well as the generation to check if other {members} are required or not. For example, if you want to know the year-on-year growth in your company:
    1) Retrieve the numbers of yearly sales using the Retriever tool
    2) Using the GPTGenericInformation tool, calculate the yearly growth with the Generation output
    If a generated answer is given, make sure that the generated answer is relevant to the question. If not, use the tools.

    If you believe the generated answer (Generation) is satisfactory and no further input is needed, respond with "FINISH" to conclude the conversation.
    """
    ) # the GPTGenericInformation team member is the preferred choice due to their broad knowledge base. 
    #"You have access to the following tools:\n\n{members}\n\nUse the following format:\n\nQuestion: the input question you must answer\nThought: you should always think about what to do\nAction: the action to take, should be one of [{members}]\nAction Input: the input to the action\nObservation: the result of the action\n... (this Thought/Action/Action Input/Observation can repeat N times)\n"
    # Our team supervisor is an LLM node. It just picks the next agent to process
    # and decides when the work is completed
    options = ["FINISH"] + members
    # Using openai function calling can make output parsing easier for us
    function_def = {
        "name": "route",
        "description": "Select the next role.",
        "parameters": {
            "title": "routeSchema",
            "type": "object",
            "properties": {
                "next": {
                    "title": "Next",
                    "anyOf": [
                        {"enum": options},
                    ],
                }
            },
            "required": ["next"],
        },
    }
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="messages"),
            (
                "system",
                "Given the conversation above, who should act next?"
                " Or should we FINISH? Select one of: {options}",
            ),
        ]
    ).partial(options=str(options), members=", ".join(members))

    #llm = ChatOpenAI(model="gpt-4-1106-preview")

    supervisor_chain = (
        prompt
        | llm.bind_functions(functions=[function_def], function_call="route")
        | JsonOutputFunctionsParser()
    )
    try:
        input=f"""
        Question:{state["keys"]["question"]}
        Generation:{state["keys"]["generation"]}
        """
    except:
        input=f"""
        Question:{state["keys"]["question"]}
        """
    #print("working till here")    
    next_step = supervisor_chain.invoke({"messages":[input]})
    
    print(f"Next Step: {next_step}")
    print(state["keys"]["question"].split("\n"))

    try:
        return {"keys": {"question": "Question: "+state["keys"]["question"].split("\n")[0]+"\n Retrived Answer from previous agent:"+state["keys"]["generation"],"generation":state["keys"]["generation"],"next_step": next_step["next"]}}
    except:
        return {"keys": {"question": state["keys"]["question"],"next_step": next_step["next"]}}
    
   
def supervisor_decision(next_step):
    """
    Supervisor decides which worker to use next

    Args:
        state (dict): The current state of the agent, including all keys.

    Returns:
        str: Next node to call
    """
    print("---SUPERVISOR DECISION---")
    if next_step['keys']['next_step'] == "FINISH":
        return 'FINISH'
    else:
        return next_step['keys']['next_step']  
    
    
def GPTGenericInformation(state):
    """
    Retrieve Generic Information

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): the information GPT model currently has
    """
    print("---Generic Model Information-----")

    fact_extraction_prompt = PromptTemplate(
    input_variables=["text_input"],
    template="You are an helpful assistant provide an detail answer for this topic and if you don't know the answer reply with 'I don't have the required knowledge for this. Please utilise other tools if required' question/query and/or generated output from previous agent Query:\n\n {text_input}"
    )
    chain = LLMChain(llm=llm, prompt=fact_extraction_prompt)

    output = chain.run(state["keys"]['question'])
    print(output)
    
    return {"keys": {"question": state["keys"]["question"],"generation": output}}

def Gmail_Cred():

    toolkit = GmailToolkit()

    # Can review scopes here https://developers.google.com/gmail/api/auth/scopes
    # For instance, readonly scope is 'https://www.googleapis.com/auth/gmail.readonly'
    credentials = get_gmail_credentials(
        token_file="token.json",
        scopes=["https://mail.google.com/"],
        client_secrets_file="credentials.json",
    )
    api_resource = build_resource_service(credentials=credentials)
    
    toolkit = GmailToolkit(api_resource=api_resource)
    
    instructions = """You are an assistant."""
    base_prompt = hub.pull("langchain-ai/openai-functions-template")
    prompt = base_prompt.partial(instructions=instructions)

    agent = create_openai_functions_agent(llm, toolkit.get_tools(), prompt)

    agent_executor = AgentExecutor(
        agent=agent,
        tools=toolkit.get_tools(),
        # This is set to False to prevent information about my email showing up on the screen
        # Normally, it is helpful to have it set to True however.
        verbose=False,
    )
    
    # agent = initialize_agent(
    # tools=toolkit.get_tools(),
    # llm=llm,
    # verbose=True,
    # agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    # )
    
    return agent_executor


agent_executor_gmail=Gmail_Cred()
 

def GmailReadWriteAgent(state):
    """
    Reading and writing mail to Gmail

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Question and Output of the mail
    """
    print("---Gmail Read Write Agent-----")

    output = agent_executor_gmail.invoke({
        "input": state["keys"]['question']
    })
    print(output)
    return {"keys": {"question": state["keys"]["question"],"generation": output['output']}}


def SlackInitialization():
    toolkit = SlackToolkit()
    tools = toolkit.get_tools()
    prompt = hub.pull("hwchase17/react")
    agent =  initialize_agent(
        tools=toolkit.get_tools(),
        llm=llm,
        verbose=False,
        agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
        )
    
    # agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=False)

    return agent
    
Slack_Agent_Executer=SlackInitialization()

def SlackAgent(state):
    """
    Reading and writing to slack channel or user group

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Question and Output of the mail
    """
    
    print("---Slack Read Write Agent-----")
    print(state["keys"]['question'])
    output=Slack_Agent_Executer.invoke(state["keys"]['question'])
    print(output)
    return {"keys": {"question": state["keys"]["question"],"generation": output['output']}}
    



import pprint
from langgraph.graph import END, StateGraph

def Graph_Compiler():
    members = ["transform_query_retriver","GPTGenericInformation","GmailReadWriteAgent","SlackAgent"]

    workflow = StateGraph(GraphState)

    # Define the nodes
    #workflow.add_node("supervisor", supervisor_main)   
    workflow.add_node("SlackAgent", SlackAgent) 
    workflow.add_node("GmailReadWriteAgent", GmailReadWriteAgent) 
    workflow.add_node("GPTGenericInformation", GPTGenericInformation) 
    workflow.add_node("supervisor", supervisor_main)  
    workflow.add_node("retrieve", retrieve)  # retrieve
    workflow.add_node("grade_documents", grade_documents)  # grade documents
    workflow.add_node("generate", generate)  # generatae
    workflow.add_node("transform_query_web", transform_query_web)
    workflow.add_node("transform_query_retriver", transform_query_retriver)# transform_query
    workflow.add_node("web_search", web_search)  # web search

        
    workflow.set_entry_point("supervisor") 

    conditional_map = {k: k for k in members}
    conditional_map["FINISH"] = END

    workflow.add_conditional_edges("supervisor", supervisor_decision, conditional_map)





    # Build graph
    #workflow.set_entry_point("retrieve")
    workflow.add_edge("transform_query_retriver", "retrieve")
    workflow.add_edge("retrieve", "grade_documents")
    workflow.add_conditional_edges(
        "grade_documents",
        decide_to_generate,
        {
            "transform_query_web": "transform_query_web",
            "generate": "generate",
        },
    )
    workflow.add_edge("transform_query_web", "web_search")
    workflow.add_edge("web_search", "generate")
    #workflow.add_edge("generate", END)
    workflow.add_edge("generate", "supervisor")
    workflow.add_edge("GPTGenericInformation", "supervisor")
    workflow.add_edge("GmailReadWriteAgent", "supervisor")
    workflow.add_edge("SlackAgent", "supervisor")
    # Compile
    app = workflow.compile()
    
    return app

app=Graph_Compiler()

