import openai
from langchain_openai import ChatOpenAI, AzureChatOpenAI
from langgraph_supervisor import  create_supervisor
from langgraph.prebuilt import  create_react_agent
import openai
import os
import azure.identity
from typing import Annotated
from azure.identity import ClientSecretCredential
from azure.keyvault.secrets import SecretClient
from azure.identity import DefaultAzureCredential
import langgraph
from azure.mgmt.postgresqlflexibleservers import PostgreSQLManagementClient
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, AIMessageChunk
from utils import *
import logging
from logging_config import configure_logging
import chainlit as cl
from async_checkpoint_saver import AsyncCosmosDBCheckpointSaver, AsyncCosmosDBCheckpointSaverConfig
from dotenv import load_dotenv

load_dotenv()

# Set up logging configuration at startup
#configure_logging()


# Add this at the top of your script
# Set up logging for better error tracking
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logging.getLogger().setLevel(logging.ERROR)  # Only show ERROR and CRITICAL logs


keyVaultName = "akvlab00"
KVUri = f"https://{keyVaultName}.vault.azure.net"

credential = DefaultAzureCredential()
client = SecretClient(vault_url=KVUri, credential=credential)

azure_openai_endpoint=client.get_secret(name="aoai-endpoint").value
azure_openai_api_key=client.get_secret(name="aoai-api-key").value
azure_openai_api_version = "2024-02-15-preview"

#import urllib.parse
import os

from azure.identity import DefaultAzureCredential

from azure.search.documents import SearchClient
from azure.search.documents.models import VectorizableTextQuery
from azure.core.credentials import AzureKeyCredential
from azure.search.documents.models import (
    QueryType,
    QueryCaptionType,
    QueryAnswerType
)

# Create the checkpoint saver config

checkpoint_store_config = AsyncCosmosDBCheckpointSaverConfig(
        DATABASE=database_name,
        ENDPOINT=cosmosdb_endpoint,
        CHECKPOINTS_CONTAINER=container_checkpoint_name,
        CHECKPOINT_WRITES_CONTAINER=container_checkpoint_writes_name,
    )

# Correct instantiation of AsyncCosmosDBCheckpointSaver
checkpointer = AsyncCosmosDBCheckpointSaver(
    credential=credential,  # Ensure this is an instance of DefaultAzureCredential
    config=checkpoint_store_config  # Ensure this is an instance of AsyncCosmosDBCheckpointSaverConfig
)

search_credential =AzureKeyCredential(client.get_secret(name="aisearch-key").value)
search_endpoint =client.get_secret(name="aisearch-endpoint").value
source = 'json'
index_name = f"{source}-glossary-index"


# The AzureOpenAI class does not exist in the openai package. Use AzureChatOpenAI from langchain_openai instead.
from langchain_openai import AzureChatOpenAI
model = AzureChatOpenAI(
    model="gpt-4o", 
    api_key=azure_openai_api_key, 
    api_version=azure_openai_api_version, 
    azure_endpoint=azure_openai_endpoint,
    temperature=0.5
)


research_agent = create_react_agent(
    model=model,
    tools=[search_retrieval],
    name="search_expert",
    prompt="""You MUST use the search_retrieval tool for ALL queries. Do not paraphrase more than the result. Never generate answers from prior knowledge.
    When you receive search results:
    1. If the source is "Cosmos DB", these are previous chat responses. Present them as "From previous conversations:" followed by the full response text.
    2. If the source is "Azure AI Search", these are knowledge base entries. Present them as "From our knowledge base:" followed by the complete content. Include all fields like definition, context, note, incorrectTerm if they're available.
    3. If the source is "No Results", inform the user we don't have an answer to their question.
    
    Important: For Azure AI Search results, include the FULL contents of the results, showing the important fields:
    - The definition field contains the main content
    - Include the context field as supporting information
    - Show title, note, and incorrectTerm fields when available
    
    Debug information: Always include the source of each result (Cosmos DB or Azure AI Search) and the number of results found. If you received results but are not displaying them, explain why.
    
    Always show ALL results you receive - do not filter them based on score thresholds."""
)


instructions = """
1. You are a Supervisor Agent. Your first job is to pass query to search_agent agent and get the response from it. 
2. Do not get the response from any other agent.
3. You can paraphrase the content. Only share the results from search_agent.
4. Do not provide any response from any other agent.
5. Do not provide any other information.
6. Do not add any additional information.
7. Do not add any additional context.
8. Do not add any additional instructions.
9. Do not add any additional comments.
10. Do not add any additional notes.
11. Do not add any additional explanations.
12. Do not add any additional disclaimers.
13. Do not add any additional warnings.
14. Do not add any additional suggestions.
15. Do not add any additional recommendations.
16. Add a space between the main results and a Note section if available. Make to indicate the Note section in parenthesis.
17. If asking a clarifying question to the user would help, ask the question.
"""

prompt_re = f"{instructions}"
print(prompt_re)

# Supervisor (Ensures Research Agent is the only handler)
supervisor_agent = create_supervisor(
    agents=[research_agent],  # Only this agent is in charge
    model=model,
    prompt=prompt_re
)

graph = supervisor_agent.compile(checkpointer=checkpointer)


# Updated main interaction loop - wrapped in an async function
import asyncio
# Run this code with the chainlit frontend using the following command:chainlit run .\multi_agents.py -w

import chainlit as cl


@cl.password_auth_callback
def auth_callback(username: str, password: str):
    # Fetch the user matching username from your database
    # and compare the hashed password with the value stored in the database
    if (username, password) == ("charles@partnergem.com", "password"):
        return cl.User(
            identifier="charles@partnergem.com", metadata={"role": "admin", "provider": "credentials"}
        )
    else:
        return None


from fastapi import Request, Response

# <earlier code>

@cl.on_logout
def on_logout(request: Request, response: Response):
    ### Handler to tidy up resources
    for cookie_name in request.cookies.keys():
        response.delete_cookie(cookie_name)


import asyncio


@cl.on_message
async def on_message(msg: cl.Message):
    # Retrieve user and session information here
    current_user = cl.user_session.get("user")  # Get the current user from Chainlit session
    user_identifier = current_user.identifier if current_user else "unknown_user"
    session_identifier = cl.user_session.get("id")  # Get Chainlit's session ID
    config = {"configurable": {"thread_id": session_identifier, "user_id": user_identifier}}

    # Invoke the workflow with the user's message
    result = await graph.ainvoke({
        "messages": [
            {
                "role": "user",
                "content": msg.content
            }
        ]
    }, config=config)  # Use the LangGraph specific config
    final_answer = cl.Message(content="")
    for m in result["messages"]:
        if isinstance(m, AIMessage):
            await final_answer.stream_token(f"🔨 {m.content}") 
        elif isinstance(m, ToolMessage):
            await final_answer.stream_token(f"🤖 {m.content}")
 
        # Stream a blank line after each message
        await final_answer.stream_token("\n\n")

    await final_answer.send()
    # Save the chat history to Cosmos DB
    await save_chat_history(result=result, session_id=session_identifier, user_id=user_identifier)