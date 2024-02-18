import logging

import pytest


@pytest.fixture(autouse=True)
def set_log_level():
    logging.getLogger().setLevel(logging.DEBUG)
