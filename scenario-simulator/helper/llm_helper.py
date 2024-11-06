import openai
from openai import OpenAI
import pandas as pd
import os
import requests
from dotenv import load_dotenv
from datetime import datetime
import uuid
from .custom_logger import CustomLogger
import json
from tqdm import tqdm
from typing import Union, List, Dict
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from enum import Enum
import torch
import psutil


class HuggingFaceModel(Enum):
    LLAMA = "meta-llama/Llama-3.1-8B-Instruct"
    GEMMA = "google/gemma-2-9b-it"
    MISTRAL = "mistralai/Mistral-Nemo-Instruct-2407"


class LLMHelper:
    """
    A helper class for interacting with LLM APIs and managing user conversations.
    """

    def __init__(self):
        """
        Initialize the LLMHelper instance.

        Examples:
            >>> llm_helper = LLMHelper()
        """
        load_dotenv()
        self.api_key = os.getenv("OPENAI_API_KEY")
        openai.api_key = self.api_key
        self.openai_client = OpenAI()
        self.logger = CustomLogger()
        self.base_path = os.path.abspath(os.path.dirname(__file__))
        self._load_config()
        self.conversations = {}  # Dictionary to store conversation history

    def _load_config(self):
        """
        Loads the configuration file to extract the endpoint for Ollama.

        The configuration file is expected to be in JSON format and contain a key called 'ollama_remote_endpoint' which stores the value of a remote Ollama endpoint for WSL to access Windows Running Ollama.
        """
        try:
            config_path = os.path.join(self.base_path, "..", "config", "config.json")
            with open(config_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.ollama_remote_endpoint = data["ollama_remote_endpoint"]
        except (FileNotFoundError, json.JSONDecodeError, KeyError) as e:
            print(f"Error: {str(e)}")

    def initialize_conversation(self, conversation_id: str = None) -> str:
        """
        Start a new conversation or retrieve an existing one.

        Args:
            conversation_id (str): An optional ID for the conversation. If not provided, a new ID will be generated.

        Returns:
            str: The ID of the conversation.

        Examples:
            >>> llm_helper = LLMHelper()
            >>> conversation_id = llm_helper.initialize_conversation()
            >>> print(conversation_id)
        """
        if not conversation_id:
            conversation_id = str(uuid.uuid4())
        if conversation_id not in self.conversations:
            self.conversations[conversation_id] = []
        return conversation_id

    def get_conversation_history(self, conversation_id: str) -> List[Dict[str, str]]:
        """
        Retrieve the conversation history for a given conversation ID.

        Args:
            conversation_id (str): The ID of the conversation.

        Returns:
            List[Dict[str, str]]: The conversation history.

        Examples:
            >>> llm_helper = LLMHelper()
            >>> conversation_id = llm_helper.initialize_conversation()
            >>> history = llm_helper.get_conversation_history(conversation_id)
            >>> print(history)
        """
        return self.conversations.get(conversation_id, [])

    def print_conversation_history(self, conversation_id: str):
        """
        Pretty-print the conversation history for a given conversation ID.

        Args:
            conversation_id (str): The ID of the conversation.

        Examples:
            >>> llm_helper = LLMHelper()
            >>> conversation_id = llm_helper.initialize_conversation()
            >>> llm_helper.print_conversation_history(conversation_id)
        """
        history = self.get_conversation_history(conversation_id)
        print(json.dumps(history, indent=4))

    def _query_openai(
        self,
        conversation_history: list,
        model_name: str,
        llm_params: dict,
        require_json_output: bool,
    ) -> str:
        """
        Query the OpenAI API.

        Args:
            conversation_history (list): The conversation history.
            model_name (str): The name of the model to use.
            llm_params (dict): Additional parameters to pass to the LLM.
            require_json_output (bool): Whether to require the response in JSON format.

        Returns:
            str: The model's response.
        """
        data = {
            "model": model_name,
            "messages": conversation_history,
        }
        data.update(llm_params)
        if require_json_output:
            data["response_format"] = {"type": "json_object"}
        response = self.openai_client.chat.completions.create(**data)
        return response.choices[0].message.content

    def _query_ollama(
        self,
        conversation_history: list,
        model_name: str,
        llm_params: dict,
        require_json_output: bool,
    ) -> str:
        """
        Query the Ollama API.

        Args:
            conversation_history (list): The conversation history.
            model_name (str): The name of the model to use.
            llm_params (dict): Additional parameters to pass to the LLM.
            require_json_output (bool): Whether to require the response in JSON format.

        Returns:
            str: The model's response.
        """
        prompt = conversation_history[-1]["content"]
        url = f"{self.ollama_remote_endpoint}/api/generate"
        data = {
            "model": model_name,
            "prompt": prompt,
            "stream": False,
            "options": llm_params,
        }
        if len(conversation_history) > 1:
            data["system"] = conversation_history[0]["content"]
        if require_json_output:
            data["format"] = "json"
        response = requests.post(url, json=data)
        return response.json()["response"]

    # For simplicty, keeping this for evaluation purposes.
    def call_llm_one_off(
        self,
        prompt,
        model_name="llama2",
        llm_params: dict = {},
        llm_response_csv_fieldname="response",
        log_file=None,
        system_prompt=None,
        require_json_output: bool = False,
        **kwargs,
    ) -> str:
        """
        Generic method to make a one-off API call to any LLM.

        Args:
            prompt (str): The user's prompt.
            model_name (str): The name of the model to use. Default is 'llama2'.
            llm_params (dict): Additional parameters to pass to LLM (i.e. temperature, etc.). Default is empty dictionary.
            llm_response_csv_fieldname (str): The CSV fieldname for LLM response. Needed because some calls are used to i.e. generate persusasive attack and naming it as 'persuasive prompts' would make more sense. Default is 'response'
            log_file (str): The file path to log the conversation details. Default is None.
            system_prompt (str): An optional system prompt to guide the conversation. Default is None.
            **kwargs: Additional keyword arguments to be logged.

        Returns:
            str: The model's response.

        Example:
            >>> llm_helper = LLMHelper()
            >>> response = llm_helper.call_llm_one_off(prompt='Who are you?',
            model_name='gpt-3.5-turbo',
            log_file='log.csv',
            system_prompt='You are a helpful assistant.')
            >>> print(response)
        """
        print(f"Q: {prompt}")
        # Check if user is running ChatGPT models
        if model_name.startswith("gpt"):
            # craft request
            url = "https://api.openai.com/v1/chat/completions"
            messages = (
                [{"role": "system", "content": system_prompt}] if system_prompt else []
            )
            messages.append({"role": "user", "content": prompt})
            data = {
                "model": model_name,
                "messages": messages,
            }
            data.update(llm_params)
            # add json output supported by OpenAI
            if require_json_output:
                data["response_format"] = {"type": "json_object"}

            # send request
            # response = self.openai_client.chat.completions.create(model=model_name, messages=data["messages"])
            response = self.openai_client.chat.completions.create(**data)
            answer = response.choices[0].message.content
        else:
            # craft request
            url = f"{self.ollama_remote_endpoint}/api/generate"
            data = {
                "model": model_name,
                "prompt": prompt,
                "stream": False,
                # Additional params for llm like temperature, seed, etc.
                # See https://github.com/ollama/ollama/blob/main/docs/api.md
                "options": llm_params,
            }
            if system_prompt:
                data["system"] = system_prompt

            if require_json_output:
                data["format"] = "json"
            # send request
            response = requests.post(url, json=data)
            answer = response.json()["response"]

        print(f"A: {answer}")
        print("")

        # Gather the default and additional details into kwargs
        kwargs.update(
            {
                "model_name": model_name,
                "system_prompt": system_prompt if system_prompt else "",
                "prompt": prompt,
                llm_response_csv_fieldname: answer,
            }
        )

        # only log if log_file parameter is supplied
        if log_file:
            self.logger.log_to_csv(log_file, **kwargs)

        return answer

    def _query_huggingface(
        self,
        conversation_history: list,
        model_enum: HuggingFaceModel,
        llm_params: dict,
        require_json_output: bool,
    ) -> str:
        """
        Queries a Hugging Face transformer model deployed across multiple GPUs.

        This function loads and configures a Hugging Face transformer model specified
        by `model_enum` to distribute computation across all available GPUs. It extracts
        the latest user message from `conversation_history` as the model prompt and
        applies the provided `llm_params` as generation parameters. For efficient memory
        management, it uses bfloat16 precision and custom memory allocation per GPU.

        Parameters:
        ----------
        conversation_history : list
            List of dictionaries representing the conversation history. The latest user
            message in the list is used as the input prompt.

        model_enum : HuggingFaceModel
            Enum specifying the Hugging Face model to be loaded. This determines model-
            specific configurations, such as special token addition or max token count.

        llm_params : dict
            Dictionary of parameters for text generation, passed directly to the Hugging
            Face pipeline (e.g., `max_new_tokens`, `temperature`, etc.).

        require_json_output : bool
            Indicates if the output should be formatted as JSON. Currently not utilized
            in this function but may be used for future output formatting.

        Returns:
        -------
        str
            The generated text response from the Hugging Face model.

        Notes:
        ------
        - The function uses `torch.cuda.device_count()` to determine available GPUs and
        prints memory allocation details for each device.
        - Model loading is configured to maximize memory utilization across GPUs using
        the `device_map` and `max_memory` parameters.
        - Configurations are applied per model type (e.g., special tokens for GEMMA,
        max token count for MISTRAL) to ensure compatibility.
        - The function uses an automatic device mapping pipeline for efficient
        multi-GPU inference.

        """
        # Extract the latest user message for querying the model
        prompt = conversation_history[-1]["content"]

        # Check available GPUs and print info
        num_gpus = torch.cuda.device_count()

        # Load the tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_enum.value)

        # Configure model loading for multi-GPU setup with proper memory formatting
        max_memory = {
            i: f"{int(torch.cuda.get_device_properties(i).total_memory * 0.85 / (1024**3))}GB"
            for i in range(num_gpus)
        }

        # Add CPU memory limit dynamically
        cpu_ram_gb = int(
            psutil.virtual_memory().total / (1024**3) * 0.85
        )  # 85% of total RAM
        max_memory["cpu"] = f"{cpu_ram_gb}GB"

        # Add CPU memory limit
        # max_memory["cpu"] = "24GB"  # Adjust this value based on your system's RAM

        # Load model with automatic device mapping
        model = AutoModelForCausalLM.from_pretrained(
            model_enum.value,
            device_map="auto",  # This will automatically distribute across GPUs
            max_memory=max_memory,  # Specify max memory per device
            torch_dtype=torch.bfloat16,  # Use bfloat16 for better memory efficiency
            low_cpu_mem_usage=True,
        )

        # Model-specific configurations
        if model_enum == HuggingFaceModel.LLAMA:
            model.config.pad_token_id = tokenizer.eos_token_id
        elif model_enum == HuggingFaceModel.GEMMA:
            special_tokens_dict = {
                "additional_special_tokens": ["<start_of_turn>", "<end_of_turn>"]
            }
            tokenizer.add_special_tokens(special_tokens_dict)
            model.resize_token_embeddings(len(tokenizer))
        elif model_enum == HuggingFaceModel.MISTRAL:
            if "max_new_tokens" not in llm_params:
                llm_params["max_new_tokens"] = 1024

        # Initialize the pipeline with automatic device mapping
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            device_map="auto",  # This will use the same device mapping as the model
            return_full_text=False,  # to avoid returning input with output
        )

        # Pass the user's params directly to the pipeline call
        response = pipe(prompt, **llm_params)

        # Extract the generated text
        answer = response[0]["generated_text"]

        return answer

    def call_llm(
        self,
        prompt: Union[str, List[str]],
        model_name: Union[str, HuggingFaceModel, List[Union[str, HuggingFaceModel]]],
        conversation_id: str = None,
        llm_params: dict = {},
        system_prompt: str = None,
        require_json_output: bool = False,
        log_file: str = None,
        **kwargs,
    ) -> Dict[str, str]:
        """
        Query LLMs with a prompt or a list of prompts and model names, allowing custom Hugging Face generation parameters.

        Args:
            prompt (Union[str, List[str]]): The user's prompt or a list of prompts.
            model_name (Union[str, HuggingFaceModel, List[Union[str, HuggingFaceModel]]]): The name of the model or a list of model names.
            conversation_id (str): An optional ID for the conversation. If not provided, a new ID will be generated.
            llm_params (dict): Additional parameters to pass to LLM. Default is an empty dictionary.
            system_prompt (str): An optional system prompt to guide the conversation. Default is None.
            require_json_output (bool): Whether to require the response to be in JSON format. Default is False.
            log_file (str): The file path to log the conversation details. Default is None.
            **kwargs: Additional keyword arguments to be logged.

        Returns:
            Dict[str, str]: A dictionary with the model names and their responses.
        """
        if isinstance(prompt, str):
            prompt = [prompt]
        if isinstance(model_name, (str, HuggingFaceModel)):
            model_name = [model_name]

        responses = {}
        conversation_id = self.initialize_conversation(conversation_id)
        conversation_history = self.conversations[conversation_id]

        for single_prompt in prompt:
            for single_model_name in model_name:
                print(f"Q ({single_model_name}): {single_prompt}")

                if system_prompt and not conversation_history:
                    conversation_history.append(
                        {"role": "system", "content": system_prompt}
                    )

                conversation_history.append({"role": "user", "content": single_prompt})

                if isinstance(single_model_name, HuggingFaceModel):
                    answer = self._query_huggingface(
                        conversation_history,
                        single_model_name,
                        llm_params,
                        require_json_output,
                    )
                elif single_model_name.startswith("gpt"):
                    answer = self._query_openai(
                        conversation_history,
                        single_model_name,
                        llm_params,
                        require_json_output,
                    )
                else:
                    answer = self._query_ollama(
                        conversation_history,
                        single_model_name,
                        llm_params,
                        require_json_output,
                    )

                conversation_history.append({"role": "assistant", "content": answer})
                self.conversations[conversation_id] = conversation_history

                print(f"A ({single_model_name}): {answer}\n")

                if single_model_name not in responses:
                    responses[single_model_name] = []

                responses[single_model_name].append(
                    {"conversation_id": conversation_id, "response": answer}
                )

                log_data = {
                    "conversation_id": conversation_id,
                    "model_name": (
                        single_model_name.value
                        if isinstance(single_model_name, HuggingFaceModel)
                        else single_model_name
                    ),
                    "prompt": single_prompt,
                    "response": answer,
                    "system_prompt": system_prompt if system_prompt else "",
                    "llm_params": json.dumps(llm_params),
                }
                log_data.update(kwargs)

                if log_file:
                    self.logger.log_to_csv(log_file, **log_data)

        return responses
