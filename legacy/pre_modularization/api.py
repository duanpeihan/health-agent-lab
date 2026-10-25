
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from health_content_agent_v3 import run_health_agent


# 1. 创建FastAPI应用
app = FastAPI(
    title="HealthAgent API",
    description="基于LangGraph的健康科普内容生成Agent",
    version="1.0.0"
)


# 2. 定义请求数据格式
class HealthRequest(BaseModel):
    task: str = Field(
        min_length=5,
        max_length=500,
        description="用户的健康科普写作任务"
    )


# 3. 健康检查接口
@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "service": "HealthAgent API"
    }


# 4. 健康科普生成接口
@app.post("/generate")
def generate_content(request: HealthRequest):

    try:
        result = run_health_agent(request.task)

        return result

    except Exception:
        # 不向API调用方暴露内部错误信息
        raise HTTPException(
            status_code=503,
            detail="Agent执行失败，请检查服务日志及本地模型"
        )
