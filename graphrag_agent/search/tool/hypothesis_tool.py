from typing import Any, Dict, List

from langchain_core.tools import BaseTool

from graphrag_agent.models.get_models import get_llm_model
from graphrag_agent.search.tool.reasoning.search import QueryGenerator
from graphrag_agent.config.prompts import SUB_QUERY_PROMPT, FOLLOWUP_QUERY_PROMPT


class HypothesisGeneratorTool:
    """生成多种假设以辅助深度研究。"""

    def __init__(self):
        self.llm = get_llm_model()
        self.query_generator = QueryGenerator(self.llm, SUB_QUERY_PROMPT, FOLLOWUP_QUERY_PROMPT)

    def generate(self, query: str) -> List[str]:
        if not query:
            raise ValueError("query不能为空")
        return QueryGenerator.generate_multiple_hypotheses(query, self.llm)

    def get_tool(self) -> BaseTool:
        generator = self

        class HypothesisTool(BaseTool):
            name: str = "hypothesis_generator"
            description: str = "假设生成工具：针对复杂问题提供2-3个可能的分析假设，帮助规划后续研究。"

            def _run(self_tool, query: Any) -> Dict[str, Any]:
                if isinstance(query, dict):
                    query = query.get("query") or query.get("input") or ""
                hypotheses = generator.generate(str(query))
                return {"query": query, "hypotheses": hypotheses}

            def _arun(self_tool, *args: Any, **kwargs: Any) -> Any:
                raise NotImplementedError("异步执行未实现")

        return HypothesisTool()
