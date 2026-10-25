
from unittest.mock import patch

from fastapi.testclient import TestClient

from api import app


client = TestClient(app)


# 测试1：健康检查接口
def test_health():

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


# 测试2：没有证据时的返回
def test_no_evidence():

    mock_result = {
        "status": "no_evidence",
        "answer": "知识库中没有找到相关资料，暂不生成内容。",
        "evidence_count": 0,
        "revision_count": 0,
        "duration_seconds": 0.01
    }

    with patch(
        "api.run_health_agent",
        return_value=mock_result
    ) as mock_agent:

        response = client.post(
            "/generate",
            json={
                "task": "请写一篇关于科学补钙的科普文章"
            }
        )

        mock_agent.assert_called_once()

    assert response.status_code == 200
    assert response.json()["status"] == "no_evidence"


# 测试3：参数校验
def test_invalid_task():

    response = client.post(
        "/generate",
        json={"task": "睡眠"}
    )

    # 输入长度小于5，应拒绝
    assert response.status_code == 422


# 测试4：内容生成成功后的API返回
def test_generate_success():

    mock_result = {
        "status": "completed",
        "answer": "成年人睡眠科普内容。[1]",
        "evidence_count": 1,
        "revision_count": 1,
        "duration_seconds": 12.5
    }

    with patch(
        "api.run_health_agent",
        return_value=mock_result
    ):

        response = client.post(
            "/generate",
            json={
                "task": "请写一篇成年人睡眠健康科普文章"
            }
        )

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "completed"
    assert data["evidence_count"] == 1
    assert data["revision_count"] == 1
    assert "睡眠" in data["answer"]
