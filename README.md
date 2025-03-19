# LLM-virtual-pretesting
Repo for the project on virtual pretesting with LLMs using in-context learning

## Activate virtual environment
```
cd "/home/abthuy/Documents/PhD research/LLM-virtual-pretesting/src"
conda activate llm_virtual_pretesting
```

## Run instructions
With configurations in e.g., `config/experiment/`:
```
python main.py experiment
```
Optional argument: `--dry-run` to predict only for 10 examples

## API keys
Create an `.env` file at the root level with contents:
```
OPENAI_API_KEY="your_openai_api_key"
```
