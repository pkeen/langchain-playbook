from langchain.prompts import ChatPromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableBranch
llm = ChatOpenAI()
import asyncio

class StreamingRunnableBranch(RunnableBranch):
    async def astream(self, input, config=None, **kwargs):
        for condition, runnable in self.branches:
            if await condition.ainvoke(input, config=config, **kwargs):
                async for output in runnable.astream(input, config=config, **kwargs):
                    yield output
                return
        # If no condition matches, stream from the default runnable
        async for output in self.default.astream(input, config=config, **kwargs):
            yield output

categorize_prompt = """
Given the following question about physics, classify it as 'beginner', 'expert' or 'uncategorized'.\n
Your answer should be based on the percieved level of the question. If it is basic (related to simpler physics topics), reply 'beginner'. If it relates to advanced physics topics reply 'expert'. If the question does not relate to physics, reply 'uncategorized'.\n
Do not respond with more than one word. \n
<question>
{question}
</question>
"""
chain = ChatPromptTemplate.from_template(template=categorize_prompt) | llm | StrOutputParser()

beginner_template="""You are a physics teacher who is really focused on beginners, explaining concepts simply. You assume no prior knowledge.
Here is your question: \n
{question}
"""
expert_template="""You are a physics professor who explains physics topics to advanced audience members. You can assume anyone you answer to has a PHD in Physics.
here is your question: \n
{question}"""

beginner_chain = ChatPromptTemplate.from_template(beginner_template) | llm
expert_chain = ChatPromptTemplate.from_template(expert_template) | llm


general_prompt = ChatPromptTemplate.from_template(
    "You are a helpful assistant. Answer the question as accurately as you can.\n\n{input}"
)
branch = StreamingRunnableBranch(
  (lambda x: "beginner" in x["topic"].lower(), beginner_chain),
  (lambda x: "expert" in x["topic"].lower(), expert_chain),
  general_prompt,
)

full_chain = {
    "topic": chain,
    "question": lambda x: x["question"]
} | branch

async def stream_output(chain, input_data):
    async for output in chain.astream(input_data):
        yield output

stream_generator = stream_output(full_chain, {"question": "what are magnets?"})

async def process_stream():
    async for s in stream_generator:
        print(s.content, end="", flush=True)

# Run the event loop to process the stream
asyncio.run(process_stream())

# This Fucking Works!