from typing import Any, Dict

from langchain_core.tools import BaseTool

from graphrag_agent.search.tool.reasoning.validator import AnswerValidator
from graphrag_agent.search.tool.hybrid_tool import HybridSearchTool


class AnswerValidationTool:
    """
    将AnswerValidator封装成LangChain Tool。
    默认复用HybridSearchTool的关键词提取能力以评估相关性。
    """

    def __init__(self):
        keyword_tool = HybridSearchTool()
        self.validator = AnswerValidator(keyword_tool.extract_keywords)

    def validate(self, query: str, answer: str) -> Dict[str, Any]:
        if not query:
            raise ValueError("query不能为空")
        if answer is None:
            answer = ""
        result = self.validator.validate(query, answer)
        return {"query": query, "answer": answer, "validation": result}

    def get_tool(self) -> BaseTool:
        validator = self

        class AnswerValidatorLC(BaseTool):
            name: str = "answer_validator"
            description: str = "答案质量验证工具：基于关键词与错误模式检测评估回答的长度、相关性与可用性。"

            def _run(self_tool, query: Any, answer: str = "", **kwargs: Any) -> Dict[str, Any]:
                payload_query = query
                payload_answer = answer
                if isinstance(query, dict):
                    payload_query = query.get("query") or query.get("input") or ""
                    payload_answer = query.get("answer") or answer
                return validator.validate(str(payload_query), str(payload_answer))

            def _arun(self_tool, *args: Any, **kwargs: Any) -> Any:
                raise NotImplementedError("异步执行未实现")

        return AnswerValidatorLC()
