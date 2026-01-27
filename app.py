"""
Chainlit app for the multi-agent marketing system.
Provides a web-based chat interface with memory and agent routing.
"""

import chainlit as cl
from chainlit import Action, AskUserMessage
import asyncio
import sys
import os
from langgraph.types import Command

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

#import graph
from autonomous import graph
import memory_setup

# http://localhost:8000
# chainlit run app.py -w -h - manual, headless login
# chainlit run app.py -w - auto login


# Password Authentication callback
# @cl.password_auth_callback
# def auth_callback(username: str, password: str):
#     """
#     Simple authentication with hardcoded user.
#     Returns user object if credentials match, None otherwise.
#     """
#     # Hardcoded credentials
#     if username == "charlesc@partnergem.com" and password == "demo123":
#         return cl.User(
#             identifier="charlesc@partnergem.com",
#             metadata={
#                 "role": "admin",
#                 "name": "Charles C",
#                 "email": "charlesc@partnergem.com"
#             }
#         )
#     return None

import httpx  # For Graph API calls

def check_user_group_membership(access_token: str, allowed_group_ids: list[str]) -> bool:
    """
    Check if the user is a member of any allowed groups using Microsoft Graph API.
    
    Args:
        access_token: OAuth access token with GroupMember.Read.All scope
        allowed_group_ids: List of allowed group GUIDs
        
    Returns:
        True if user is in at least one allowed group, False otherwise
    """
    if not allowed_group_ids or not allowed_group_ids[0]:
        print("No allowed groups configured - allowing all users")
        return True
    
    try:
        # Call Graph API to check group membership
        response = httpx.post(
            "https://graph.microsoft.com/v1.0/me/checkMemberGroups",
            headers={
                "Authorization": f"Bearer {access_token}",
                "Content-Type": "application/json"
            },
            json={"groupIds": allowed_group_ids},
            timeout=10.0
        )
        
        if response.status_code == 200:
            member_groups = response.json().get("value", [])
            print(f"User is member of groups: {member_groups}")
            return len(member_groups) > 0
        else:
            print(f"Graph API error: {response.status_code} - {response.text}")
            return False
            
    except Exception as e:
        print(f"Error checking group membership: {e}")
        return False


# OAuth Authentication callback
@cl.oauth_callback
def oauth_callback(provider_id: str, token: str, raw_user_data: dict, default_user: cl.User) -> cl.User | None:
    """
    Handle OAuth callback from Azure AD.
    raw_user_data contains user profile from Microsoft Graph API.
    
    Graph API field mapping:
    - userPrincipalName: user's email/UPN
    - displayName: user's full name
    - id: Azure AD object ID
    - mail: user's email (may be None)
    - jobTitle, givenName, surname: additional profile info
    """
    # DEBUG: Print what we received
    print(f"=== OAuth Callback Debug ===")
    print(f"Provider: {provider_id}")
    print(f"Raw user data: {raw_user_data}")
    print(f"============================")
    
    # Extract user identifier (use userPrincipalName from Graph API)
    user_identifier = (
        raw_user_data.get("userPrincipalName") or 
        raw_user_data.get("mail") or 
        raw_user_data.get("preferred_username") or
        raw_user_data.get("email")
    )
    
    if not user_identifier:
        print("ERROR: No user identifier found in response")
        return None
    
    print(f"User identifier: {user_identifier}")
    
    # Check group membership using Graph API
    allowed_groups = os.getenv("ALLOWED_GROUP_IDS", "").split(",")
    allowed_groups = [g.strip() for g in allowed_groups if g.strip()]
    
    if allowed_groups:
        if not check_user_group_membership(token, allowed_groups):
            print(f"ACCESS DENIED - User {user_identifier} not in allowed groups")
            return None
    
    print("ACCESS GRANTED")
    
    # Return authorized user with Graph API field names
    return cl.User(
        identifier=user_identifier,
        metadata={
            "name": raw_user_data.get("displayName") or raw_user_data.get("name"),
            "email": raw_user_data.get("mail") or raw_user_data.get("userPrincipalName"),
            "job_title": raw_user_data.get("jobTitle"),
            "oid": raw_user_data.get("id") or raw_user_data.get("oid"),
        }
    )


@cl.on_chat_start
async def start():
    """Initialize the chat session"""
    # Setup containers on first run
    try:
        await memory_setup.setup_checkpoint_containers()
    except Exception as e:
        await cl.Message(content=f"⚠️ Container setup warning: {str(e)}").send()
    
    # Get authenticated user (guaranteed to exist after login)
    user = cl.user_session.get("user")
    user_id = user.identifier  # charlesc@partnergem.com
    thread_id = cl.user_session.get("id")
    
    cl.user_session.set("config", {
        "configurable": {
            "thread_id": thread_id,
            "user_id": user_id
        }
    })
    # Personalized welcome message with job title
    user_name = user.metadata.get("name", user_id.split("@")[0].title())
    job_title = user.metadata.get("job_title")
    
    # Build greeting based on available info
    if job_title:
        greeting = f"👋 Welcome back, **{user_name}** ({job_title})!"
    else:
        greeting = f"👋 Welcome back, **{user_name}**!"
    
    await cl.Message(
        content=f"{greeting}\n\n"
                "I can help you with:\n"
                "- 📊 **Data analysis** and SQL queries (via Genie)\n"
                "- 🔍 **Web searches** for current information\n"
                "- 💾 **Remember and recall** information from our conversations\n\n"
                "What can I help you with today?"
    ).send()


@cl.on_message
async def main(message: cl.Message):
    """Handle incoming messages with HITL support"""
    config_dict = cl.user_session.get("config")
    workflow_graph = graph.get_graph()
    
    # Show thinking indicator
    async with cl.Step(name="Processing", type="llm") as step:
        step.input = message.content
        
        # Invoke the graph
        payload = {"messages": [{"role": "user", "content": message.content}]}
        result = await workflow_graph.ainvoke(payload, config=config_dict)
        
        step.output = "Processing complete"
    
    # Check for interrupt (single check - no while loop needed)
    if "__interrupt__" in result:
        interrupt_data = result["__interrupt__"][0].value
        
        # Store interrupt data and result in session for callback access
        cl.user_session.set("pending_interrupt", {
            "interrupt_data": interrupt_data,
            "config": config_dict
        })
        
        # Get the message from interrupt data
        warning_message = interrupt_data.get("message", "⚠️ **Review Required**")
        query = interrupt_data.get("query", "")
        
        # Show response with approval actions
        actions = [
            cl.Action(name="approve_action", value="approve", label="✓ Approve", payload={"action": "approve"}),
            cl.Action(name="reject_action", value="reject", label="✗ Cancel", payload={"action": "reject"})
        ]
        
        await cl.Message(
            content=f"{warning_message}\n\n**Query:** {query}",
            actions=actions
        ).send()
        return  # Wait for action callback
    
    # Extract final response
    if result["messages"]:
        last_message = result["messages"][-1]
        response_content = last_message.content
        agent_name = getattr(last_message, "name", "Assistant")
    else:
        response_content = "I encountered an issue processing your request."
        agent_name = "System"
    
    # Show execution metrics if available
    exec_times = result.get("execution_times", {})
    if exec_times:
        agent = list(exec_times.keys())[-1]
        duration = exec_times[agent]
        metadata = f"⏱️ {agent}: {duration:.2f}s"
    else:
        metadata = ""
    
    # Determine agent icon based on agent name
    agent_icons = {
        "Genie": "📊",
        "Search": "🔍",
        "Memory": "💾"
    }
    agent_icon = agent_icons.get(agent_name, "🤖")
    
    # Send final response with clear agent identification
    await cl.Message(
        content=f"{agent_icon} **{agent_name} Agent**\n\n{response_content}",
        metadata={"info": metadata} if metadata else None
    ).send()


@cl.action_callback("approve_action")
async def on_approve(action: cl.Action):
    """Handle approve action"""
    pending = cl.user_session.get("pending_interrupt")
    if not pending:
        await cl.Message(content="Error: No pending operation to approve.").send()
        return
    
    workflow_graph = graph.get_graph()
    config_dict = pending["config"]
    
    await cl.Message(content="✅ Operation approved. Executing...").send()
    
    # Resume workflow with approval
    result = await workflow_graph.ainvoke(
        Command(resume=True),
        config=config_dict
    )
    
    # Clear pending interrupt
    cl.user_session.set("pending_interrupt", None)
    
    # Extract and display final response
    if result["messages"]:
        last_message = result["messages"][-1]
        response_content = last_message.content
        agent_name = getattr(last_message, "name", "Assistant")
        await cl.Message(content=f"📊 **{agent_name} Agent**\n\n{response_content}").send()
    else:
        await cl.Message(content="Operation completed.").send()


@cl.action_callback("reject_action")
async def on_reject(action: cl.Action):
    """Handle reject action"""
    pending = cl.user_session.get("pending_interrupt")
    if not pending:
        await cl.Message(content="Error: No pending operation to cancel.").send()
        return
    
    workflow_graph = graph.get_graph()
    config_dict = pending["config"]
    
    await cl.Message(content="❌ Operation cancelled.").send()
    
    # Resume workflow with cancellation
    result = await workflow_graph.ainvoke(
        Command(resume=None),
        config=config_dict
    )
    
    # Clear pending interrupt
    cl.user_session.set("pending_interrupt", None)
    
    # Show cancellation message
    if result.get("messages"):
        last_message = result["messages"][-1]
        await cl.Message(content=last_message.content).send()


@cl.on_chat_end
async def end():
    """Handle chat session end"""
    await cl.Message(content="Thanks for chatting! Your conversation has been saved.").send()


if __name__ == "__main__":
    # This allows running the app with: python -m chainlit run app.py
    from chainlit.cli import run_chainlit
    run_chainlit(__file__)
