import pytest

import synth


@pytest.fixture
def synth_data(tmp_path):
    """tmp_path 아래에 시험용 가상 자료 한 벌을 쓰고 경로와 설정을 담은 사전을 돌려준다."""
    return synth.write_all(tmp_path)
