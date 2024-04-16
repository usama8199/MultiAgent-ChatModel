# MultiAgent-ChatModel
We have created a multi agent chat model which utilizes multiple agents to perform various task

## Introduction
We have created multi agent chat model using which you can perform various task be it mailing to a person or browsing something from web. We have utilize [Langchain](https://python.langchain.com/docs/get_started/quickstart/) and [Langgraph](https://python.langchain.com/docs/langgraph/) which extends the LangChain Expression Language with the ability to coordinate multiple chains (or actors) across multiple steps of computation in a cyclic manner. It is inspired by [Pregel](https://research.google/pubs/pregel-a-system-for-large-scale-graph-processing/) and [Apache Beam](https://beam.apache.org/). The current interface exposed is one inspired by [NetworkX](https://networkx.org/documentation/latest/). 





#### Examples
1. If i wanted to know who won women ipl 2024 which is not possible using LLM as it will not have latest knowledge so it will utilize web browser tool to extract the information and provide a suitable answetr
2. If i wanted to extract the total and due date of a person from a document and mail it to them i can ustilize rag(Retrival Augmented Generation) agent
