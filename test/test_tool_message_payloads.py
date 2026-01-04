import unittest

from langchain_core.messages import AIMessage, ToolMessage

from graphrag_agent.agents.graph_agent import GraphAgent
from graphrag_agent.agents.hybrid_agent import HybridAgent


class _DummyTool:
    def __init__(self, result):
        self.result = result
        self.last_query = None

    def search(self, query):
        self.last_query = query
        return self.result


class ToolMessagePayloadTests(unittest.IsolatedAsyncioTestCase):
    async def test_graph_agent_tool_message_uses_string_content_for_dict(self):
        agent = GraphAgent.__new__(GraphAgent)
        dummy_tool = _DummyTool({"answer": "dict response"})
        agent.local_tool = dummy_tool
        agent.global_tool = dummy_tool

        state = {
            "messages": [
                AIMessage(
                    content="",
                    additional_kwargs={
                        "tool_calls": [
                            {
                                "id": "call-1",
                                "function": {
                                    "name": "local_retriever",
                                    "arguments": {"query": "hello"},
                                },
                            }
                        ]
                    },
                )
            ]
        }

        result = await agent._retrieve_node_async(state)

        self.assertIn("messages", result)
        self.assertEqual(len(result["messages"]), 1)
        tool_message = result["messages"][0]

        self.assertIsInstance(tool_message, ToolMessage)
        self.assertIsInstance(tool_message.content, str)
        self.assertEqual(tool_message.content, "dict response")
        self.assertEqual(dummy_tool.last_query, "hello")

    async def test_hybrid_agent_tool_message_preserves_string_payload(self):
        agent = HybridAgent.__new__(HybridAgent)
        agent.search_tool = _DummyTool("string payload")

        state = {
            "messages": [
                AIMessage(
                    content="",
                    additional_kwargs={
                        "tool_calls": [
                            {
                                "id": "call-2",
                                "function": {
                                    "name": "search_tool",
                                    "arguments": {"query": "world"},
                                },
                            }
                        ]
                    },
                )
            ]
        }

        result = await agent._retrieve_node_async(state)

        self.assertIn("messages", result)
        self.assertEqual(len(result["messages"]), 1)
        tool_message = result["messages"][0]

        self.assertIsInstance(tool_message, ToolMessage)
        self.assertIsInstance(tool_message.content, str)
        self.assertEqual(tool_message.content, "string payload")
        self.assertEqual(agent.search_tool.last_query, "world")


if __name__ == "__main__":
    unittest.main()
