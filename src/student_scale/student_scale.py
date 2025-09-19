"""Module for system prompt."""

# standard library imports
# /

# related third party imports
import inflect

# local application/library specific imports
from student_scale.build import STUDENT_SCALE_REGISTRY


@STUDENT_SCALE_REGISTRY.register("digits_int")
def build_digits_int(num_groups: int) -> tuple[dict[str, str], str]:
    student_levels_base = list(range(1, num_groups + 1))
    mapping = {
        str(i): str(i) for i in student_levels_base
    }  # mapping digits as strings to themselves
    list_string = (
        f"(with {str(student_levels_base[0])} as the lowest level "
        f"and {str(student_levels_base[-1])} as the highest level)"
    )
    return mapping, list_string


@STUDENT_SCALE_REGISTRY.register("digits_str")
def build_digits_str(num_groups: int) -> tuple[dict[str, str], str]:
    student_levels_base = list(range(1, num_groups + 1))
    p = inflect.engine()
    mapping = {
        str(i): p.number_to_words(i) for i in student_levels_base
    }  # mapping digits as strings to themselves
    list_string = (
        f"(with {mapping[str(student_levels_base[0])]} as the lowest level "
        f"and {mapping[str(student_levels_base[-1])]} as the highest level)"
    )
    return mapping, list_string


@STUDENT_SCALE_REGISTRY.register("american")
def build_american(num_groups: int) -> tuple[dict[str, str], str]:
    if num_groups != 5:
        raise ValueError("The 'american' student scale is only defined for 5 groups.")
    student_levels_base = list(range(1, num_groups + 1))
    student_levels_american = ["F", "D", "C", "B", "A"]  # NOTE: reverse order!
    mapping = {str(i): student_levels_american[i - 1] for i in student_levels_base}
    list_string = (
        f"(with {mapping[str(student_levels_base[0])]} as the lowest level "
        f"and {mapping[str(student_levels_base[-1])]} as the highest level)"
    )
    return mapping, list_string


@STUDENT_SCALE_REGISTRY.register("proficiency_3_str")
def build_proficiency_3_str(num_groups: int) -> tuple[dict[str, str], str]:
    if num_groups != 3:
        raise ValueError(
            "The 'proficiency_3_str' student scale is only defined for 3 groups."
        )
    student_levels_base = list(range(1, num_groups + 1))
    student_levels = ["Beginner", "Intermediate", "Advanced"]  # NOTE: reverse order!
    mapping = {str(i): student_levels[i - 1] for i in student_levels_base}
    list_string = (
        f"(with {mapping[str(student_levels_base[0])]} as the lowest level "
        f"and {mapping[str(student_levels_base[-1])]} as the highest level)"
    )
    return mapping, list_string


@STUDENT_SCALE_REGISTRY.register("proficiency_5_str")
def build_proficiency_5_str(num_groups: int) -> tuple[dict[str, str], str]:
    if num_groups != 5:
        raise ValueError(
            "The 'proficiency_5_str' student scale is only defined for 5 groups."
        )
    student_levels_base = list(range(1, num_groups + 1))
    student_levels = [
        "1 (Fundamental Awareness)",
        "2 (Novice)",
        "3 (Intermediate)",
        "4 (Advanced)",
        "5 (Expert)",
    ]  # NOTE: reverse order!
    mapping = {str(i): student_levels[i - 1] for i in student_levels_base}
    list_string = ", ".join(
        [level for _, level in enumerate(student_levels)]
    )
    list_string = f"(of levels {list_string})"
    return mapping, list_string
