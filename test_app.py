import streamlit as st
import asyncio
import aiohttp
import uuid

# Initialize session state for chat history and user_id
if "userid" not in st.session_state:
    st.session_state.userid = str(uuid.uuid4())

if "messages" not in st.session_state:
    st.session_state.messages = []

# Function to interact with the backend API
async def mog_query(user_id: str, chat_history: list, query: str, source: str = "website"):
    url = "http://localhost:8020/api/chat"
    headers = {
        "accept": "application/json",
        "Content-Type": "application/json"
    }
    payload = {
        "userId": user_id,
        "chat_history": chat_history,
        "user_query": query,
        "source": source
    }

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=headers, json=payload) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    return {"error": f"HTTP {response.status}: {await response.text()}"}
    except Exception as e:
        return {"error": str(e)}

def main():
    st.title("Sales Bot Simulation Test UI")
    st.markdown("This is a simplified UI specifically meant for browser automation testing.")

    # Display chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Chat input
    if prompt := st.chat_input("Type your message here..."):
        # Show user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Prepare history for API (excluding the current prompt)
        api_history = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages]
        
        # Add to local history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Call API
        with st.chat_message("assistant"):
            with st.spinner("Bot is thinking..."):
                response_data = asyncio.run(mog_query(
                    user_id=st.session_state.userid,
                    chat_history=api_history,
                    query=prompt
                ))
            
            if "response" in response_data:
                bot_reply = response_data["response"]
                st.markdown(bot_reply)
                st.session_state.messages.append({"role": "assistant", "content": bot_reply})
            else:
                st.error(f"Error: {response_data.get('error', 'Unknown error')}")

if __name__ == "__main__":
    main()
