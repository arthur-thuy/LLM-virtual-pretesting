# LLM-virtual-pretesting
Repo for the project on virtual pretesting with LLMs using in-context learning

## Run instructions

## Short

Activate the virtual environment:
```bash
cd "/home/abthuy/Documents/PhD research/LLM-virtual-pretesting"  # your own path
source .venv/bin/activate
```

With configurations in e.g., `config/experiment/`:
```bash
cd src
python main_replication.py experiment
```
Optional argument: `--dry-run` to predict only for 10 examples
> NOTE: in order to save the logs to a file, append `2>&1 | tee output/log` to the command above.

## Detailed
More detailed instructions are available in in [`CONTRIBUTING.md`](CONTRIBUTING.md).

## API keys
Create an `.env` file at the root level with contents:
```
OPENAI_API_KEY="..."
LANGFUSE_SECRET_KEY="..."
LANGFUSE_PUBLIC_KEY="..."
LANGFUSE_HOST="..."
PINECONE_API_KEY="..."
ANTHROPIC_API_KEY="..."
GOOGLE_API_KEY="..."
```

## Links

[Langfuse dashboard](https://cloud.langfuse.com/project/cm8n8clg300k7ad07l3pjqklk)

[Pinecone indexes](https://app.pinecone.io/organizations/-OMHNPCneLFsq5lU1w0X/projects/82bcf4da-9c8e-43ac-8c41-43e4fc58d8d3/indexes?sessionType=login)


