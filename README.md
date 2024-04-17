# MultiAgent-ChatModel

## Introduction
We have created a multi-agent chat model that allows you to perform various tasks, such as sending emails to people or browsing the web. We have utilized [Langchain](https://python.langchain.com/docs/get_started/quickstart/) and [Langgraph](https://python.langchain.com/docs/langgraph/), which extends the LangChain Expression Language with the ability to coordinate multiple chains (or actors) across multiple steps of computation in a cyclic manner. It is inspired by [Pregel](https://research.google/pubs/pregel-a-system-for-large-scale-graph-processing/) and [Apache Beam](https://beam.apache.org/).

The current interface exposed is inspired by [NetworkX](https://networkx.org/documentation/latest/). It utilizes the [React](https://arxiv.org/pdf/2210.03629.pdf) method, which will figure out the action and, once completed, reflect on that result to determine if it solves the given task. This process will repeat until the given problem is solved. In our case, it will utilize multiple agents to solve the given task.


## Architecture 
The image below shows the high-level design of the architecture. A supervisor will understand the task and utilize one or more agents, depending on whether it can answer or solve the desired task. Once an agent completes its task, it will provide the result to the [supervisior](https://github.com/langchain-ai/langgraph/blob/main/examples/multi_agent/agent_supervisor.ipynb), which will decide whether the task is completed or if another tool is required. Regardless of the situation, the chain will come back to the supervisor. Below, you can see some examples, video explanations, and a demo.

### High Level Design
<img src="https://github.com/usama8199/MultiAgent-ChatModel/blob/main/Image/Overview.png" width="400" height="500"/>

#### Examples
1. If you want to know who won the Women's IPL 2024, which is not possible using an LLM (Large Language Model) as it will not have the latest knowledge, the system will utilize a web browser tool to extract the information and provide a suitable answer.
2. If you want to extract the total amount and due date for a person from a document and email it to them, you can utilize a RAG (Retrieval Augmented Generation) agent to extract the user's total and due date, then use GPT to construct an email with the total and due date, and finally, utilize a mail agent to send the email to that person.

#### Video Explanation (Click On the Image)
[<img src="https://img.youtube.com/vi/tC5P0R3mBJs/maxresdefault.jpg" width="70%">](https://youtu.be/tC5P0R3mBJs)


## CRAG (inside one of the agent)
[CRAG](https://arxiv.org/pdf/2401.15884.pdf) is a method that utilizes re-ranking to evaluate the retrieved documents or chunks and uses the most relevant chunks based on semantic and syntactic meaning

<img src="https://github.com/usama8199/MultiAgent-ChatModel/blob/main/Image/CRAG.png" width="1400" height="250"/>

**Steps**
1. First, question transformation occurs, which makes the question more RAG-friendly. For example, changing "RNN" to "Recurrent Neural Networks".
2. Then, it will extract the relevant documents using the question and some similarity search methods.
3. It will then utilize the question and retrieved documents or chunks, and use the most relevant chunks based on semantic and syntactic meaning using GPT models and an appropriate prompt (in the paper, they have trained a T5 model).
4. If no relevant chunks are found, the chain will search the web for relevant information, provide an answer, and send it to the supervisor.
5. If even one relevant chunk is found, it will use that to generate an answer and send it to the supervisor, which will decide the next steps.

# Setup

1. Git clone the repository
2. Install the dependency using `pip install -r requirements.txt`
3. Create a .env variable and add all the keys
    1. TAVILY_API_KEY can be found [here](https://docs.tavily.com/docs/gpt-researcher/getting-started)
    2. AZURE_OPENAI_API_KEY and AZURE_OPENAI_ENDPOINT will be given by organization or you can also use just open api key (you just need to change AzureChatOpenAI and AzureOpenAIEmbeddings to openai one which you can find online in langgraph_crag_Main.py)
    3. LANGCHAIN_API_KEY for langsmith can be found [here](https://smith.langchain.com/)
    4. SLACK_BOT_TOKEN can be found in [slack app](https://api.slack.com/tutorials/tracks/getting-a-token)
4. You also need to have gmail Credentials.json which can be found [here](https://www.youtube.com/watch?v=_pZebYlgGcY)
5. You can then finally run the `streamlit run streamlit_ui.py`

Note: If you are not able to get gmail credentials you can still run it it's just you will not be able to call gmail agent


# Applications
1. Information extraction from invoice data, send as a report on slack to relevant users or stakeholders, in one go.
2. Retrieve relevant information from the confluence or a database without needing to store the information somewhere and get quick answers from the model.
3. Perform multiple step processes for engineers from planning to utilizing various API to complete the task, reflect and improvise on it.

# Reference
. https://python.langchain.com/docs/langgraph \
. https://smith.langchain.com/ \
. https://arxiv.org/pdf/2401.15884.pdf \
. https://python.langchain.com/docs/integrations/toolkits 

