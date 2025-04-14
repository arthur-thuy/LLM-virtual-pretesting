# Running a basic experiment

1. Clone the repository locally:
```bash
$ git clone git@github.com:your_name_here/LLM-virtual-pretesting.git
$ cd src
```
2. Download the processed datasets from the Google Drive folder and paste them in `data/processed/`.
3. Create a `.env` file at the root level with the following contents (see `example.env`):
```bash
OPENAI_API_KEY="..."
LANGFUSE_SECRET_KEY="..."
LANGFUSE_PUBLIC_KEY="..."
LANGFUSE_HOST="..."
PINECONE_API_KEY="..."
ANTHROPIC_API_KEY="..."
```
4. Create and ctivate your virtual environment:
```bash
conda env create -f environment.yml
conda activate llm_virtual_pretesting
```
5. Create a new branch from `main`:
```bash
$ git checkout main
$ git branch fix_bug
$ git checkout fix_bug
```
6. Create a new experiment in e.g., `config/experiment_kate/`. This should hold a `config.py` file and at least one yaml file. See `config/experiment/` for an example. The `config.py` file has all the defaults; each yaml file can override the defaults.
7. Run the experiment:
```bash
python main.py experiment_kate
```
> Optional argument: `--dry-run` to predict only for 10 examples.

8. The results are stored in `output/experiment_kate_<date>/`.
9. Inspect the results in `analysis.ipynb` by filling in the appropriate `EXP_NAME` value (see the output folder for the exact name with timestamp).
10. To inspect the prompt structure, refer to the [Langfuse dashboard](https://cloud.langfuse.com/project/cm8n8clg300k7ad07l3pjqklk).


# Implementing changes

If you want to change any of the building blocks, e.g., example selector, prompt structure, structured JSON output, etc, do the following.

Let's say yo want to add and use a new example selector.
1. Go to `src/example_selector/example_selector.py`.
2. Add a new class that inherits from `BaseExampleSelector`, e.g., `StudentIDRandomExampleSelector` (see LangChain documentation).
3. Add a new function that calls the class with the appropriate parameters. For example,
```python
@EXAMPLE_SELECTOR_REGISTRY.register("studentid_random")
def build_studentid_random(cfg: CfgNode, examples: list[dict]) -> BaseExampleSelector:
    input_vars = ["student_id"]
    selector = StudentIDRandomExampleSelector(
        examples=examples,
        k=cfg.EXAMPLE_SELECTOR.NUM_EXAMPLES,
    )
    return (selector, input_vars)
```
> In this case, LangChain needs the input variables that the example selector uses.

4. In your configuration folder, change the `config.py` or a yaml file to use this new building block.
5. As usual, run with `python main.py experiment_kate` and inspect the results in `analysis.ipynb`.






