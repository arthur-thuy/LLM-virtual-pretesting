"""Module for system prompt."""

# standard library imports
# /

# related third party imports
# /

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
