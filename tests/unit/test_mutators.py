#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""pytests for :mod:`recumpiler.mutators`"""


import pytest

from recumpiler.mutators import fuck_text_blob


@pytest.mark.timeout(60)
@pytest.mark.parametrize("text", [
    "foo bar",
    "Hey there guys today I'm making a cool python script that recompiles text into a more unreadable heap of garbage. Let me know what you guys think of it!",
    "crush my cock with a rock i must, maximum pain i must endure. okay here we go *crumpushh* oooooooooooooooooooooooooooooooooOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA",
    "hello",
])
def test_mutate_text_blob(text):
    out_str = fuck_text_blob(text)
    assert out_str
    print(out_str)
