"""Each dataclass-to-argparse mapping row, especially the two bool branches."""
import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional

import pytest

from augenblick.core.config import add_dataclass_arguments, config_from_namespace


@dataclass(frozen=True)
class Sample:
    flag_off: bool = False
    flag_on: bool = True
    count: int = 7
    ratio: float = 0.5
    label: str = "hi"
    maybe: Optional[int] = None
    choice: Literal["a", "b"] = "a"
    iters: list[int] = field(default_factory=lambda: [1, 2])
    where: Path = Path("/tmp")
    renamed: int = field(default=3, metadata={"cli_name": "--other-name"})


def parser_for(cls):
    p = argparse.ArgumentParser()
    add_dataclass_arguments(p, cls)
    return p


def test_defaults_round_trip():
    cfg = config_from_namespace(Sample, parser_for(Sample).parse_args([]))
    assert cfg == Sample()


def test_bool_false_is_store_true():
    ns = parser_for(Sample).parse_args(["--flag_off"])
    assert ns.flag_off is True


def test_bool_true_gets_negated_flag():
    ns = parser_for(Sample).parse_args(["--no-flag_on"])
    assert ns.flag_on is False


def test_numeric_and_str_types():
    ns = parser_for(Sample).parse_args(["--count", "9", "--ratio", "1.5", "--label", "x"])
    assert (ns.count, ns.ratio, ns.label) == (9, 1.5, "x")


def test_optional_parses_inner_type():
    ns = parser_for(Sample).parse_args(["--maybe", "4"])
    assert ns.maybe == 4


def test_literal_restricts_choices():
    assert parser_for(Sample).parse_args(["--choice", "b"]).choice == "b"
    with pytest.raises(SystemExit):
        parser_for(Sample).parse_args(["--choice", "z"])


def test_list_uses_nargs_plus():
    assert parser_for(Sample).parse_args(["--iters", "3", "4", "5"]).iters == [3, 4, 5]


def test_path_field():
    assert parser_for(Sample).parse_args(["--where", "/x"]).where == Path("/x")


def test_cli_name_override():
    ns = parser_for(Sample).parse_args(["--other-name", "8"])
    assert ns.renamed == 8


def test_config_from_namespace_ignores_extras():
    ns = parser_for(Sample).parse_args([])
    ns.unrelated = "ignored"
    assert config_from_namespace(Sample, ns) == Sample()
