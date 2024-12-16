# technical-exercise-dec-16-2024

## Overview

- This repository implements TextRank summarization and GPT-3.5-turbo to summarize a corpus of text.
- If you want to use GPT-3.5-turbo, you need to set the `OPENAI_API_KEY` environment variable in the `.env` file.


## Setup

1. Clone the repository
2. Install the dependencies
3. Set the environment variables and `config.yaml` file
4. Run the code

## Install the dependencies

Create a virtual/conda/pip environment and install the dependencies using `pip`. Python 3.12 is recommended.

```bash
pip install -r requirements.txt
```

## Configuration

- Set the parameters in the `src/config.yaml` file: 
- Set the `OPENAI_API_KEY` environment variable in the `.env` file (if you want to use GPT-3.5-turbo)

## Running the code

First navigate to the `src` directory and run the code using

```bash
python extract_requirements.py
```

The following files will be created:
- `results/extracted_requirements.csv`
- `logs/logs.txt`

## Testing

Run the tests using 

```bash
pytest
```


