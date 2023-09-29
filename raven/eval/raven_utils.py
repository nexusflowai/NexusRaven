from typing import List, Union

from langchain.agents import Tool, AgentOutputParser
from langchain.schema import AgentAction, AgentFinish, OutputParserException
from langchain.prompts import StringPromptTemplate


RAVEN_PROMPT = """
{raven_tools}
User Query: Question: {input}

Please pick a function from the above options that best answers the user query and fill in the appropriate arguments.<human_end>"""


def construct_raven_prompt(raven_tools):
    return RAVEN_PROMPT.format(raven_tools=raven_tools)


# Set up a prompt template
class RavenPromptTemplate(StringPromptTemplate):
    # The template to use
    template: str
    # The list of tools available
    tools: List[Tool]

    def format(self, **kwargs) -> str:
        prompt = "<human>:\n"
        for tool in self.tools:
            func_signature, func_docstring = tool.description.split(" - ", 1)
            prompt += f'\nOPTION:\n<func_start>def {func_signature}<func_end>\n<docstring_start>\n"""\n{func_docstring}\n"""\n<docstring_end>\n'
        kwargs["raven_tools"] = prompt
        return self.template.format(**kwargs).replace("{{", "{").replace("}}", "}")


class RavenOutputParser(AgentOutputParser):
    def parse(self, text: str) -> Union[AgentAction, AgentFinish]:
        # Check if agent should finish
        if not "Initial Answer:" in text:
            raise OutputParserException(f"Could not parse LLM output: `{text}`")

        output = text.strip().split("\n")[1].replace("Initial Answer: ", "").strip()
        return AgentFinish(return_values={"output": output}, log=text)
