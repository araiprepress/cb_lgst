
from langchain.agents import ConversationalChatAgent, AgentExecutor, Tool
from langchain.tools.base import ToolException

from langchain.callbacks import StreamlitCallbackHandler
from langchain.chains import LLMChain, LLMMathChain

from langchain.chat_models import ChatOpenAI

from langchain.utilities import DuckDuckGoSearchAPIWrapper
from langchain.tools import WikipediaQueryRun
from langchain.utilities import WikipediaAPIWrapper

from langchain.memory import ConversationBufferMemory
from langchain.memory.chat_message_histories import StreamlitChatMessageHistory
import streamlit as st

st.set_page_config(page_title="StreamlitChatMessageHistory & langchain agent", page_icon="📖")
st.title("📖 StreamlitChatMessageHistory & langchain_agent")

"""
comment here.
[source code for this app]( https://github.com/araiprepress/cb_lgst ).
"""

#  openai api keys ===========
user_openai_api_key = st.sidebar.text_input(
    "OpenAI API Key", type="password", help="input OpenAI API Key."
)

if not user_openai_api_key:
    st.sidebar.warning("Enter an OpenAI API Key to continue")
    st.stop()

#  memory and history ===========
msgs = StreamlitChatMessageHistory(key="langchain_messages")
memory = ConversationBufferMemory(
    chat_memory=msgs, return_messages=True, memory_key="chat_history", output_key="output"
)

clear_btn = st.sidebar.button("Reset chat history")

if len(msgs.messages) == 0 or clear_btn:
    msgs.clear()
    msgs.add_ai_message("How can I help you?")
    st.session_state.steps = {}

avatars = {"human": "😊", "ai": "🤖"}
for idx, msg in enumerate(msgs.messages):
    with st.chat_message(avatars[msg.type]):
        # Render intermediate steps if any were saved
        for step in st.session_state.steps.get(str(idx), []):
            if step[0].tool == "_Exception":
                continue
            with st.status(f"{step[0].tool}: {step[0].tool_input}", state="complete"):
                st.write(step[0].log)
                st.write(step[1])
        st.write(msg.content)

#  llm chain ===========
llm = ChatOpenAI(model_name="gpt-3.5-turbo",openai_api_key=user_openai_api_key, streaming=True)
# llm = OpenAI(temperature=0, openai_api_key=user_openai_api_key, streaming=True)
# llm_chain = LLMChain(llm=llm, prompt=prompt, memory=memory)

# tools' funcs =============
search = DuckDuckGoSearchAPIWrapper()
# llm_math_chain = LLMMathChain.from_llm(llm)
wikipedia_search = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())


def llm_math_chain_func(llm):
    try:
        LLMMathChain.from_llm(llm=llm, verbose=True).run
    except ZeroDivisionError as e:
        return str('ZeroDivisionError')
    except Exception as e:
        # print(e)
        return str('llm_math_chain error')

# def wiki_search():
#     try:
#         WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
#     except Exception as e:
#         return e
def wiki_search(query: str):
    if query == "":
        # query = st.chat_input()
        query = st.session_state.get("last_query")
    try:
        run = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
        return run(query)
    except Exception as e:
        raise ToolException(e)

tools = [
    Tool(
        name="Search",
        func=search.run,
        description="useful for when you need to answer questions about current events. You should ask targeted questions",
    ),
    Tool(
        name="Calculator",
        func=llm_math_chain_func,
        description="useful for when you need to answer questions about math. Return ZeroDivisionError when zero division. When and other errors return llm_math_chain error. ",
    ),
    Tool(
        name="WikipediaSearch",
        # func=wikipedia_search,
        # func=wiki_search(query=""),
        func=wiki_search,
        description="Searches Wikipedia for the given query and returns the summary of the top result.",
        handle_tool_error="Wiki search error" # propagate
    ),
]

def _handle_error(error) -> str:
    return str(error)[:50]

# Initialize agent =====
app_agent = ConversationalChatAgent.from_llm_and_tools(llm=llm, tools=tools)
executor = AgentExecutor.from_agent_and_tools(
        agent=app_agent,
        tools=tools,
        # agent_kwargs=agent_kwargs,
        memory=memory,
        handle_parsing_errors=_handle_error,
        return_intermediate_steps=True,
    )

if user_query := st.chat_input():
    st.chat_message("user", avatar="😊").write(user_query)
    # print(prompt)  #: What is Wiki? / Who is Japanese Prime Minister? ... eg.
    # Note: new messages are saved to history automatically by Langchain during run
    with st.chat_message("assistant", avatar="🤖"):
        st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
        response = executor(user_query, callbacks=[st_cb])
        st.write(response['output'])
        st.session_state.steps[str(len(msgs.messages) - 1)] = response["intermediate_steps"]
