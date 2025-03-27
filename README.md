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

LANGFUSE_SECRET_KEY="..."
LANGFUSE_PUBLIC_KEY="..."
LANGFUSE_HOST="..."
```

# Links

[Langfuse dashboard](https://cloud.langfuse.com/project/cm8n8clg300k7ad07l3pjqklk)

[Pineconde indexes](https://app.pinecone.io/organizations/-OMHNPCneLFsq5lU1w0X/projects/82bcf4da-9c8e-43ac-8c41-43e4fc58d8d3/indexes?sessionType=login)
