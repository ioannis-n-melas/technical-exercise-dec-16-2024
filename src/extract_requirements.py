import os
import logging
import yaml
import pandas as pd
from dotenv import load_dotenv

from lib import read_corpus, split_into_sections, simulate_llm_summary

if __name__ == "__main__":

    # read the config file
    # check if the config file exists
    if not os.path.exists("config.yaml"):
        raise FileNotFoundError("config.yaml file not found")
    with open("config.yaml", "r") as file:
        config = yaml.safe_load(file)

    # Load environment variables from the .env file
    # check if the .env file exists, if not create it
    if not os.path.exists(config["env_file"]):
        with open(config["env_file"], "w") as file:
            file.write("# .env file for the project\n")
    load_dotenv(dotenv_path=config["env_file"])


    # initialize the logger
    logging.basicConfig(filename=os.path.join(config["logs_dir"], config["logs_file"]), level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # log the config
    logging.info(f"Config: {config}")

    # read the corpus from the file
    logging.info(f"Reading corpus from {os.path.join(config['data_dir'], config['corpus_file'])}")
    corpus = read_corpus(os.path.join(config["data_dir"], config["corpus_file"]))

    # split the corpus into sections
    logging.info(f"Splitting corpus into sections")
    sections = split_into_sections(corpus, config["delimiter"])
    logging.info(f"Number of sections: {len(sections)}")

    # simulate the llm summaries
    logging.info(f"Simulating LLM summaries")
    summaries = []
    for i, section in enumerate(sections):
        summary, compression_ratio = simulate_llm_summary(section, config["llm_model"])
        summaries.append([i, section, summary, compression_ratio])

    # write the summaries to a csv file using pandas
    logging.info(f"Writing summaries to {os.path.join(config['results_dir'], config['summaries_file'])}")
    df = pd.DataFrame(summaries, columns=["Section", "Original Text", "Summarized Requirements", "Compression Ratio"])
    df.to_csv(os.path.join(config["results_dir"], config["summaries_file"]), index=False)

    logging.info(f"Summaries written successfully")

