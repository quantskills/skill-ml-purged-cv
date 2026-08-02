"""Deterministic general CPCV Path Decomposition."""

from __future__ import annotations

from collections.abc import Iterable

from .domain import (
    CPCVPath,
    CPCVPathDecomposition,
    CPCVPathOccurrence,
    TestBlock,
)


def build_cpcv_path_decomposition(
    *,
    groups: tuple[TestBlock, ...],
    combinations: tuple[tuple[int, ...], ...],
) -> CPCVPathDecomposition:
    """Properly edge-color the combination/group incidence graph."""

    n_groups = len(groups)
    n_test_groups = len(combinations[0])
    occurrences_by_group = [
        sum(group_index in combination for combination in combinations)
        for group_index in range(n_groups)
    ]
    path_count = max(occurrences_by_group)
    edge_endpoints = tuple(
        (combination_index, group_index)
        for combination_index, combination in enumerate(combinations)
        for group_index in combination
    )
    edge_colors: dict[int, int] = {}
    colors_at_combination: dict[tuple[int, int], int] = {}
    colors_at_group: dict[tuple[int, int], int] = {}

    for edge_index, (combination_index, group_index) in enumerate(edge_endpoints):
        missing_at_combination = {
            color
            for color in range(path_count)
            if (combination_index, color) not in colors_at_combination
        }
        missing_at_group = {
            color
            for color in range(path_count)
            if (group_index, color) not in colors_at_group
        }
        common = missing_at_combination & missing_at_group
        if common:
            color = min(common)
        else:
            combination_missing = min(missing_at_combination)
            group_missing = min(missing_at_group)
            component = _alternating_component(
                start_combination=combination_index,
                first_color=combination_missing,
                second_color=group_missing,
                edge_endpoints=edge_endpoints,
                colors_at_combination=colors_at_combination,
                colors_at_group=colors_at_group,
            )
            _swap_component_colors(
                component,
                first_color=combination_missing,
                second_color=group_missing,
                edge_endpoints=edge_endpoints,
                edge_colors=edge_colors,
                colors_at_combination=colors_at_combination,
                colors_at_group=colors_at_group,
            )
            color = group_missing
        edge_colors[edge_index] = color
        colors_at_combination[(combination_index, color)] = edge_index
        colors_at_group[(group_index, color)] = edge_index

    paths = tuple(
        CPCVPath(
            path_index=path_index,
            occurrences=tuple(
                sorted(
                    (
                        CPCVPathOccurrence(combination_index, group_index)
                        for edge_index, (combination_index, group_index) in enumerate(
                            edge_endpoints
                        )
                        if edge_colors[edge_index] == path_index
                    ),
                    key=lambda item: item.group_index,
                )
            ),
        )
        for path_index in range(path_count)
    )
    return CPCVPathDecomposition(
        n_groups=n_groups,
        n_test_groups=n_test_groups,
        groups=groups,
        combinations=combinations,
        paths=paths,
    )


def _alternating_component(
    *,
    start_combination: int,
    first_color: int,
    second_color: int,
    edge_endpoints: tuple[tuple[int, int], ...],
    colors_at_combination: dict[tuple[int, int], int],
    colors_at_group: dict[tuple[int, int], int],
) -> set[int]:
    stack: list[tuple[str, int]] = [("combination", start_combination)]
    visited: set[tuple[str, int]] = set()
    component: set[int] = set()
    while stack:
        side, vertex = stack.pop()
        node = (side, vertex)
        if node in visited:
            continue
        visited.add(node)
        color_map = colors_at_combination if side == "combination" else colors_at_group
        for color in (first_color, second_color):
            edge_index = color_map.get((vertex, color))
            if edge_index is None:
                continue
            component.add(edge_index)
            combination_index, group_index = edge_endpoints[edge_index]
            stack.append(
                ("group", group_index)
                if side == "combination"
                else ("combination", combination_index)
            )
    return component


def _swap_component_colors(
    component: Iterable[int],
    *,
    first_color: int,
    second_color: int,
    edge_endpoints: tuple[tuple[int, int], ...],
    edge_colors: dict[int, int],
    colors_at_combination: dict[tuple[int, int], int],
    colors_at_group: dict[tuple[int, int], int],
) -> None:
    component_edges = tuple(component)
    for edge_index in component_edges:
        combination_index, group_index = edge_endpoints[edge_index]
        old_color = edge_colors[edge_index]
        del colors_at_combination[(combination_index, old_color)]
        del colors_at_group[(group_index, old_color)]
    for edge_index in component_edges:
        combination_index, group_index = edge_endpoints[edge_index]
        old_color = edge_colors[edge_index]
        new_color = second_color if old_color == first_color else first_color
        edge_colors[edge_index] = new_color
        colors_at_combination[(combination_index, new_color)] = edge_index
        colors_at_group[(group_index, new_color)] = edge_index
