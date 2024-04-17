# MultiAgent-ChatModel
We have created a multi agent chat model which utilizes multiple agents to perform various task

## Introduction
We have created multi agent chat model using which you can perform various task be it mailing to a person or browsing something from web. We have utilize [Langchain](https://python.langchain.com/docs/get_started/quickstart/) and [Langgraph](https://python.langchain.com/docs/langgraph/) which extends the LangChain Expression Language with the ability to coordinate multiple chains (or actors) across multiple steps of computation in a cyclic manner. It is inspired by [Pregel](https://research.google/pubs/pregel-a-system-for-large-scale-graph-processing/) and [Apache Beam](https://beam.apache.org/). The current interface exposed is one inspired by [NetworkX](https://networkx.org/documentation/latest/). It utilize [React](https://arxiv.org/pdf/2210.03629.pdf) method which will figure out the action and once completed reflect on that result if that solves the given task. This will repeat again and again until it solves the given problem. In our case it will utlize multiple agents to solves the given task. Examples will be given in the susequest section


## Architecture 
The below image shows the high level design of the architecture. Here there will be [supervisior](https://github.com/langchain-ai/langgraph/blob/main/examples/multi_agent/agent_supervisor.ipynb) which will understand the task and utlize one or more agents depending on weather it answer or solves the desired task. Once the agent task is completed it will provide the result to the supervisor which will decide weather the task is completed or any other tools is required. So no matter what the chain will comes back to supervisior. Below you can see some examples and video explnation

##### Examples
1. If i wanted to know who won women ipl 2024 which is not possible using LLM as it will not have latest knowledge so it will utilize web browser tool to extract the information and provide a suitable answer
2. If i wanted to extract the total and due date of a person from a document and mail it to them i can ustilize rag(Retrival Augmented Generation) agent and extract the user total and due date and then utilize gpt to costruct a mail with total and due date then utilize mail agent to mail to that person

### High Level Design
<img src="https://github.com/usama8199/MultiAgent-ChatModel/blob/main/Image/Overview.png" width="400" height="500"/>

## CRAG (inside one of the agent)
[CRAG](https://arxiv.org/pdf/2401.15884.pdf) is a method which utilizes re-ranking to evaluates the retrive documents or chunks and use the most relevent chunks on semantic and syntatic meaning. Below chain is created using langgraph


<img src="https://github.com/usama8199/MultiAgent-ChatModel/blob/main/Image/CRAG.png" width="1400" height="250"/>

**Steps**
1. First the question transformation happens which is more rag friendly for eg changing RNN to Recurrent Neural Networks
2. Then it will extract the document using question using some similarity search methods
3. It will then utilize the question and retrive documents or chunks and use the most relevent chunks based on semantic and syntatic meaning using GPT models and appropriate prompt (in paper they have train a T5 model)
4. If we don't find any relevent chunks, chain will go on web to search the relevent information, provide answer and send that to supervisior
5. If even one relevent chunk was found it will use that to generate an answer and send that to supervisior which will decide the next steps
