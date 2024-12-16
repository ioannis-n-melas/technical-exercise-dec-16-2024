import pandas as pd
import os
import yaml
import logging
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.text_rank import TextRankSummarizer
from langchain.chains import LLMChain
from dotenv import load_dotenv

from langchain_community.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate


def textrank_summary(text_section: str) -> str:
    """
    Generate a summary using the TextRank algorithm.
    Args:
        text_section (str): The text section to summarize.
    Returns:
        str: The summary of the text section.
    """
    # Parse the text
    parser = PlaintextParser.from_string(text_section, Tokenizer("english"))
    
    # Initialize the TextRank summarizer
    summarizer = TextRankSummarizer()
    
    # Generate the summary
    summary = summarizer(parser.document, 2)  # Number of sentences in summary
    
    return " ".join([str(sentence) for sentence in summary])

def read_corpus(file_path: str) -> str:
    """
    Read the corpus from the file.
    Args:
        file_path (str): The path to the corpus file.
    Returns:
        str: The corpus text.
    """
    # check if the file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"{file_path} file not found")
    with open(file_path, "r") as file:
        return file.read()
    

def simulate_llm_summary(text_section: str, model_name: str = "TextRank") -> tuple[str, float]:
    """
    Simulate the LLM summary using LangChain with a transformer model.
    Args:
        text_section (str): The text section to summarize.
        model_name (str): The name of the model to use. Valid options are "TextRank" and "gpt-3.5-turbo".
    Returns:
        str: The summary of the text section.
        float: The compression ratio of the summary.
    """
    if model_name == "TextRank":
        summary = textrank_summary(text_section)

        # calculate compression ratio
        compression_ratio = len(text_section) / len(summary)

        return summary, compression_ratio

    elif model_name == "gpt-3.5-turbo":

        prompt_template = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful assistant."),
            ("user", "Summarize the following text: {text}")
        ])
        # check if the api key is set
        if os.getenv('OPENAI_API_KEY') is None:
            raise ValueError("OPENAI_API_KEY is not set")

        # get the api key from the environment variables
        api_key = os.getenv('OPENAI_API_KEY')

        # initialize the llm with the specified model and api key
        llm = ChatOpenAI(api_key=api_key, model_name=model_name)

        # create an llm chain with the prompt template and llm
        chain = LLMChain(llm=llm, prompt=prompt_template)

        # generate the summary
        summary = chain.run(text=text_section)

        # calculate compression ratio
        compression_ratio = len(text_section) / len(summary)

        return summary, compression_ratio   

    else:
        raise ValueError(f"Invalid model name: {model_name}")
    

def split_into_sections(corpus: str, delimiter: str = "\n\n") -> list[str]:
    """
    Split the corpus into sections.
    Args:
        corpus (str): The corpus text.
    Returns:
        list[str]: The list of sections.
    """
    return corpus.split(delimiter)


