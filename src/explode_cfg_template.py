"""Module for generating experiment configurations from a template YAML file."""

# standard library imports
import argparse
import yaml
import itertools
import os
import shutil
from pathlib import Path

# related third party imports
import structlog

# local application/library specific imports
from tools.utils import delete_previous_content, ensure_dir

# set up logger
logger = structlog.get_logger(__name__)

parser = argparse.ArgumentParser(description="Run experiment")
parser.add_argument(
    "config",
    type=str,
    help="config file path",
)


def flatten_dict(d, parent_key="", sep=".") -> dict:
    """Flatten a nested dictionary by joining keys with separator."""
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def unflatten_dict(d, sep=".") -> dict:
    """Convert a flattened dictionary back to nested structure."""
    result = {}
    for key, value in d.items():
        parts = key.split(sep)
        d = result
        for part in parts[:-1]:
            if part not in d:
                d[part] = {}
            d = d[part]
        d[parts[-1]] = value
    return result


def generate_experiment_configs(template_path: str, output_dir: str) -> None:
    """Generate experiment configurations from a template YAML.

    Parameters
    ----------
    template_path : str
        Path to the template YAML file.
    output_dir : str
        Directory to save the generated configurations.
    """
    # Read the template file
    with open(template_path, "r") as f:
        template = yaml.safe_load(f)

    # Flatten the dictionary
    flat_template = flatten_dict(template)

    # Find all keys with list values
    list_keys = [k for k, v in flat_template.items() if isinstance(v, list)]

    # Extract lists of values for each key
    lists = [flat_template[k] for k in list_keys]

    # Generate all combinations
    combinations = list(itertools.product(*lists))

    # Create output directory if it doesn't exist
    ensure_dir(output_dir)

    # Generate a config file for each combination
    for i, combination in enumerate(combinations):
        # Create a new config dictionary
        config = dict(flat_template)

        # Replace list values with single values from the combination
        for key, value in zip(list_keys, combination):
            config[key] = value

        # Unflatten the dictionary
        nested_config = unflatten_dict(config)

        # Save to a new YAML file
        output_path = os.path.join(output_dir, f"experiment_{i+1}.yaml")
        with open(output_path, "w") as f:
            yaml.dump(nested_config, f, default_flow_style=False)

    # copy the config.py file to the output directory
    config_py_path = os.path.join(os.path.dirname(template_path), "config.py")
    new_config_py_path = os.path.join(output_dir, "config.py")
    shutil.copy(config_py_path, new_config_py_path)

    logger.info(
        "Generated experiment configurations",
        number=len(combinations),
        output_dir=output_dir,
    )


def check_dir_name(dir_name: str) -> str:
    """Check if the directory is valid for explosion.

    Parameters
    ----------
    dir_name : str
        Directory name to check.

    Returns
    -------
    str
        Directory name without "_template".

    Raises
    ------
    ValueError
        If the directory name does not end with "_template".
    ValueError
        If the directory does not exist.
    ValueError
        If the directory does not contain exactly one YAML file named "template.yaml".
    ValueError
        If the directory does not contain a "config.py" file.
    """
    # check if name ends in "_template"
    if not dir_name.endswith("_template"):
        raise ValueError(f"Directory name '{dir_name}' does not end with '_template'.")
    # check if dir exists
    if not os.path.isdir(dir_name):
        raise ValueError(f"Directory '{dir_name}' does not exist.")
    # check if there is only 1 yaml file in the directory,
    # and that is has name "template.yaml"
    yaml_files = list(Path(dir_name).glob("*.yaml"))
    if len(yaml_files) != 1 or yaml_files[0].name != "template.yaml":
        raise ValueError(
            f"Directory '{dir_name}' does not contain exactly one YAML file "
            "named 'template.yaml'."
        )
    # check if there is a config.py file in the directory
    if not os.path.isfile(os.path.join(dir_name, "config.py")):
        raise ValueError(f"Directory '{dir_name}' does not contain a 'config.py' file.")
    # check if dirname without "_template" exists
    new_dir_name = dir_name[: -len("_template")]
    # delete previous content if it exists
    delete_previous_content(new_dir_name)
    return new_dir_name


def main() -> None:
    """Run explosion of config files."""
    args = parser.parse_args()
    config_dir = os.path.join("config", args.config)

    # check if config dir is valid
    output_dir = check_dir_name(config_dir)

    # path to the template file
    template_path = os.path.join(config_dir, "template.yaml")

    # create the new configuration files
    generate_experiment_configs(template_path, output_dir)


if __name__ == "__main__":
    main()
