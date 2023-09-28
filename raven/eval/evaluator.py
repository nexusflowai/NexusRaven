from typing import Any, Callable, Dict, List

from dataclasses import dataclass

import os

import json

from os import environ

from string import ascii_lowercase

import argparse

from unittest.mock import MagicMock

from datasets import Dataset, load_dataset

from langchain.tools.base import StructuredTool
from langchain.llms import OpenAI, HuggingFaceTextGenInference, OpenAIChat
from langchain.chat_models import ChatOpenAI
from langchain.agents import (
    AgentExecutor,
    AgentType,
    LLMSingleActionAgent,
    initialize_agent,
)
from langchain.chains import LLMChain

from raven import CACHE_DIR
from raven.utils import build_functions, parse_function_call_to_name_and_args
from raven.eval.raven_utils import (
    RavenOutputParser,
    RavenPromptTemplate,
    RAVEN_PROMPT,
)


@dataclass
class Evaluator:
    task_name: str
    llm_name: str
    agent_name: str
    hf_path: str
    standardized_queries_subset: str
    standardized_api_list_subset: str
    inference_server_url: str | None = None

    def run(self) -> None:
        functions = self.build_functions()

        llm = self.build_llm()
        tools = {
            function_name: StructuredTool.from_function(function, return_direct=True)
            for function_name, function in functions.items()
        }
        agent = self.build_agent(llm, list(tools.values()))

        """
        Currently there's not really a good way of retrieving the actual prompt sent to the model
        by LangChain (maybe through CallbackRunManager in the future?)
        so here we monkeypatch the `LLMChain.prep_prompts` method and grab the output
        """
        prompts = []
        if isinstance(agent, AgentExecutor):
            print(
                f"Monkeypatching `prep_prompts` to collect the final prompt sent by `{self.agent_name}` to `{self.llm_name}`"
            )
            original_prep_prompts = LLMChain.prep_prompts

            def monkey_patched_prep_prompts(*args, **kwargs):
                output = original_prep_prompts(*args, **kwargs)
                prompts.append(output[0][0].to_string())
                return output

            LLMChain.prep_prompts = monkey_patched_prep_prompts

        eval_dataset = self.get_eval_dataset()
        predictions = []
        references = []
        accuracies = []

        for i, sample in enumerate(eval_dataset):
            context_tools = [tools[k] for k in sample["context_functions"]]
            agent = self.build_agent(llm, context_tools)

            prompt = sample["prompt"]
            error_message = None
            output = None
            original_output = None
            try:
                original_output = agent.run(prompt)
                if isinstance(original_output, tuple):  # Successful call
                    output = original_output
                elif isinstance(
                    original_output, str
                ):  # Model returned execution string
                    function_name, args, kwargs = parse_function_call_to_name_and_args(
                        original_output
                    )
                    output = functions[function_name](*args, **kwargs)
                else:
                    raise ValueError(
                        f"Unknown output type `{original_output}` of type {type(original_output)}"
                    )
            except Exception as e:  # pylint: disable=broad-exception-caught
                error_message = f"Original output\n\t{original_output}\nOutput\n\t{output}\nError message\n\t{str(e)}\n\n"
                output = (None, None)

            predicted_function_name, predicted_args_dict = output
            reference_function_name = sample["python_function_name"]
            reference_input_args_dict = json.loads(sample["python_args_dict"])
            _, reference_args_dict = functions[reference_function_name](
                **reference_input_args_dict
            )

            function_name_match = predicted_function_name == reference_function_name
            args_dict_match = predicted_args_dict == reference_args_dict
            accuracy = function_name_match and args_dict_match

            accuracy_str = "CORRECT" if accuracy else "WRONG"
            error_message_str = "" if error_message is None else error_message
            prompt_str = prompts[-1] if prompts else prompt
            print(
                f"Example {i+1} / {len(eval_dataset)}\n\nPrompt\n\t{prompt_str}\n\nReference\n\t{reference_function_name}, {reference_args_dict}\n\nPrediction\n\t{predicted_function_name}, {predicted_args_dict}\n\n{accuracy_str}\n\n{error_message_str}{'-' * 80}\n"
            )

            if error_message is not None:
                predictions.append(error_message)
            else:
                predictions.append(output)
            references.append((reference_function_name, reference_args_dict))
            accuracies.append(accuracy)

        accuracy = 100 * sum(accuracies) / len(eval_dataset)
        print(f"Accuracy: {accuracy:.3f}%")

        predictions_d = Dataset.from_dict(
            {
                "prompt": prompts or eval_dataset["prompt"],
                "prediction": list(map(json.dumps, predictions)),
                "reference": list(map(json.dumps, references)),
                "correct": accuracies,
            }
        )
        subset = f"{self.llm_name}---{self.agent_name}"
        predictions_hf_path = f"{self.hf_path}_predictions"
        split = self.task_name
        save_path = os.path.join(CACHE_DIR, predictions_hf_path, subset, split)
        predictions_d.save_to_disk(save_path)
        print(
            f"""
You can inspect the predictions afterwards by using

```
from datasets import load_from_disk

path = "{save_path}"
dataset = load_from_disk(path)
```
"""
        )

    def get_eval_dataset(self) -> Dataset:
        d = load_dataset(
            path=self.hf_path,
            name=self.standardized_queries_subset,
            split="train",
        )
        d = d.filter(
            lambda dataset: dataset == self.task_name,
            input_columns="dataset",
        )
        assert (
            len(d) > 0
        ), f"Unknown task `{self.task_name}`. Available tasks: {list(dict.fromkeys(d['dataset']))}"

        return d

    def build_llm(self) -> Any:
        if "openai" in self.llm_name:
            api_key = environ.get("OPENAI_API_KEY", None)
            if api_key is None:
                api_key = input(
                    f"Please input your OpenAI API key to use the `{self.llm_name}` model: "
                )
                environ["OPENAI_API_KEY"] = api_key

        gpt_params = {
            "temperature": 0,
            "verbose": True,
        }
        hf_params = {
            "max_new_tokens": 400,
            "do_sample": False,
            "inference_server_url": self.inference_server_url,
            "temperature": 0.001,
        }
        match self.llm_name:
            case "openai_gpt4":
                return OpenAIChat(**gpt_params, model_name="gpt-4")
            case "chat_openai_gpt4":
                return ChatOpenAI(**gpt_params, model_name="gpt-4")
            case "openai_gpt3.5":
                return OpenAIChat(**gpt_params, model_name="gpt-3.5-turbo-16k")
            case "chat_openai_gpt3.5":
                return ChatOpenAI(**gpt_params, model_name="gpt-3.5-turbo-16k")
            case "openai_gpt3.5_instruct":
                return OpenAI(**gpt_params, model_name="gpt-3.5-turbo-instruct")
            case "nexusraven" | "prototype":
                return HuggingFaceTextGenInference(**hf_params)
            case _:
                return self.llm_name

    def build_functions(self) -> Dict[str, Callable]:
        d = load_dataset(
            path=self.hf_path,
            name=self.standardized_api_list_subset,
            split="train",
        )
        d = d.filter(
            lambda dataset: dataset == self.task_name,
            input_columns="dataset",
        )
        return build_functions(d)

    def build_agent(self, llm: Any, tools: List[StructuredTool]) -> Any:
        match self.agent_name:
            case "OPENAI_FUNCTIONS":
                return initialize_agent(
                    tools,
                    llm,
                    agent=AgentType.OPENAI_FUNCTIONS,
                    verbose=True,
                    max_iterations=1,
                )
            case "SIMPLE":
                return initialize_agent(
                    tools,
                    llm,
                    agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
                    verbose=True,
                    max_iterations=1,
                )
            case "SIMPLE_NONCHAT":
                return initialize_agent(
                    tools,
                    llm,
                    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                    verbose=True,
                    max_iterations=1,
                )
            case "NEXUSRAVEN":
                raven_prompt = RavenPromptTemplate(
                    template=RAVEN_PROMPT, tools=tools, input_variables=["input"]
                )
                llm_chain = LLMChain(llm=llm, prompt=raven_prompt)
                output_parser = RavenOutputParser()
                agent = LLMSingleActionAgent(
                    llm_chain=llm_chain,
                    output_parser=output_parser,
                    stop=["\nReflection:"],
                    allowed_tools=tools,
                )
                agent_chain = AgentExecutor.from_agent_and_tools(
                    agent=agent, tools=tools, verbose=True
                )
                return agent_chain
            case "TOOLLLM":
                return self._build_toolllm_agent(llm, tools)
            case "TOOLALPACA":
                return self._build_toolalpaca_agent(llm, tools)
            case _:
                raise KeyError(
                    f"Invalid agent_name `{self.agent_name}`. Available agent_name's: {self.available_agents}"
                )

    def _build_toolllm_agent(self, llm: Any, tools: List[StructuredTool]) -> Any:
        if hasattr(self, "agent"):
            return self.agent

        responses = load_dataset(
            self.hf_path,
            name=llm,
            split="train",
        )

        prompt_to_response = dict()
        for response in responses["response"]:
            prompt = response[0]["query"]
            response = response[1]["function_call"]
            prompt_to_response[prompt] = response

        agent = MagicMock()

        def run(prompt: str) -> str:
            return prompt_to_response.get(prompt, None)

        agent.run = run

        self.agent = agent
        return agent

    def _build_toolalpaca_agent(self, llm: Any, tools: List[StructuredTool]) -> Any:
        """
        TODO this is currently hardcoded to a specific dataset path
        """

        if hasattr(self, "agent"):
            return self.agent

        responses = load_dataset(
            "Nexusflow/toolalpaca_eval",
            split="train",
        )
        responses = responses.filter(
            lambda s: s == llm, input_columns="toolalpaca_model"
        )

        def to_upper(s: str) -> str:
            for c in ascii_lowercase:
                s = s.replace(f"_{c}", c.upper())
            return s

        if self.task_name in ["virustotal", "emailrep"]:
            to_upper = lambda s: s

        prompt_to_response = dict()
        for d in responses:
            prompt = d["prompt"]

            output = d["toolalpaca_response"]
            if len(output) == 0:
                continue

            function_name, args_dict = output[0]
            function_name = function_name.removeprefix("Action: ")
            function_name_mapping = {
                "search_cve": "searchCVE",
                "search_cpe": "searchCPE",
            }
            function_name = function_name_mapping.get(function_name, function_name)

            args_dict = args_dict.removeprefix("Action Input: ")
            try:
                args_dict = eval(args_dict)
            except:  # Model output is not a valid python object
                continue

            args_strs = []
            for arg_name, arg_value in args_dict.items():
                if isinstance(arg_value, str):
                    arg_value_str = f'"{arg_value}"'
                else:
                    arg_value_str = arg_value

                arg_name = to_upper(arg_name)
                arg_str = f"{arg_name}={arg_value_str}"
                args_strs.append(arg_str)

            args_str = ", ".join(args_strs)

            function_str = f"{function_name}({args_str})"

            prompt_to_response[prompt] = function_str

        agent = MagicMock()

        def run(prompt: str) -> str:
            return prompt_to_response.get(prompt, None)

        agent.run = run

        self.agent = agent

        return agent


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_name", type=str, required=True)
    parser.add_argument("--llm_name", type=str, required=True)
    parser.add_argument("--agent_name", type=str, required=True)
    parser.add_argument("--hf_path", type=str, required=True)
    parser.add_argument("--standardized_queries_subset", type=str, required=True)
    parser.add_argument("--standardized_api_list_subset", type=str, required=True)
    parser.add_argument("--inference_server_url", type=str)
    args = parser.parse_args()

    Evaluator(**vars(args)).run()
