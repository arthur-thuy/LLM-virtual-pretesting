# Running a basic experiment

1. Clone the repository locally:
```bash
$ git clone git@github.com:your_name_here/LLM-virtual-pretesting.git
$ cd src
```
2. Download the processed datasets from the [Google Drive](https://drive.google.com/drive/u/1/folders/1RuDHku2xI1Y3cdxk9HcpMn0BRAlDGXbo) folder and paste them in `data/processed/`.
3. Create a `.env` file at the root level with the following contents (see `example.env`):
```bash
OPENAI_API_KEY="..."
LANGFUSE_SECRET_KEY="..."
LANGFUSE_PUBLIC_KEY="..."
LANGFUSE_HOST="..."
PINECONE_API_KEY="..."
ANTHROPIC_API_KEY="..."
GOOGLE_API_KEY="..."
```
4. Create and activate your virtual environment:
```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
```
5. Create a new branch from `main`:
```bash
$ git checkout main
$ git branch fix_bug
$ git checkout fix_bug
```
6. Create a new experiment in e.g., `src/config/experiment_kate/`. This should hold a `config.py` file and at least one yaml file. See `src/config/experiment/` for an example. The `config.py` file has all the defaults; each yaml file can override the defaults.
7. Run the experiment:
```bash
cd src
python main_replication.py experiment_kate
```
> Optional argument: `--dry-run` to predict only for 10 examples.

8. The results are stored in `output/experiment_kate_<date>/`.
9. Inspect the results in `analysis_replication.ipynb` by filling in the appropriate `EXP_NAME` value (see the output folder for the exact name with timestamp).
10. To inspect the prompt structure, refer to the [Langfuse dashboard](https://cloud.langfuse.com/project/cm8n8clg300k7ad07l3pjqklk).


# Handling a large number of configurations

If you want to run a large number of configurations, it is cumbersome to create them manually. You can create them automatically as follows:
1. Create a new configuration folder that ends in `"_template"`, e.g., `config/kate_auto_template`.
2. Create a `config.py` file in this folder that holds the default values for the parameters you want to change.
3. Create a new yaml file `template.yaml` in this folder where you define a list for each parameter you want to change. For example:
```yaml
MODEL:
  NAME: ["llama3", "gpt-4o", "olmo2:7b"]

EXAMPLE_SELECTOR:
  NAME: ["random", "studentid_random", "studentid_semantic", "studentid_recency"]
```

4. Run the script `explode_cfg_template.py` in the `src` folder. This will create a new configuration folder for each combination of parameters in the template. The new folder will be named `kate_auto` (so without the `"_template"` suffix). Run as follows:
```bash
python explode_cfg_template.py kate_auto_template
```
5. You can now run the `kate_auto` experiment as usual:
```bash
python main_replication.py kate_auto
```

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
5. As usual, run with `python main_replication.py experiment_kate` and inspect the results in `analysis_replication.ipynb`.
6. Commit the changes and push to your remote branch:
```bash
$ git add src/example_selector/example_selector.py
$ git commit -m "Add new example selector"
$ git push origin fix_bug
```

# Roleplaying experiments

1. Create a new experiment in e.g., `config/roleplay_kate/`. This should hold a `config.py` file and at least one yaml file. See `config/roleplay/` for an example. The `config.py` file has all the defaults; each yaml file can override the defaults. The `config.py` file should include the following::
```python
_C.ROLEPLAY = CN()
# number of student levels to simulate
_C.ROLEPLAY.NUM_STUDENT_LEVELS = 5
# student level scale
_C.ROLEPLAY.STUDENT_SCALE = "american"
```
> These configurations are also required for the student replication experiments to work, but they are not used.
The `_C.EXAMPLE_SELECTOR.NAME` and `_C.PROMPT.NAME` should be specific for roleplaying experiments, e.g., `studentlevel_random` and `roleplay_teacher_A`.

2. Run the experiment:
```bash
cd src
python main_roleplay.py roleplay_kate
```
> Optional argument: `--dry-run` to predict only for 2 questions, so 2*NUM_STUDENT_LEVELS LLM calls.

3. The results are stored in `output/roleplay_kate_<date>/`.
4. Inspect the results in `analysis_roleplay.ipynb` by filling in the appropriate `EXP_NAME` value (see the output folder for the exact name with timestamp).





