from typing import Any, Optional, cast
import logging

# core imports (migrated from legacy)
from llama_index.core.tools import QueryEngineTool

from src.Llama_index_sandbox.custom_react_agent.tools.tool_output import CustomToolOutput


class CustomQueryEngineTool(QueryEngineTool):
    def call(self, input: Any) -> CustomToolOutput:
        query_str = cast(str, input)
        logging.info("Starting synchronous query engine tool")  #  Query with content: {query_str}")

        # TODO 2023-12-08: we'll have to tear apart the methods triggered
        #  by this function to perform node reranking
        try:
            response = self._query_engine.query(query_str)
            logging.info(f"Received synchronous response from query engine tool: {response}")
        except Exception as e:
            logging.error(f"Synchronous query from query engine tool failed: {e}", exc_info=True)
            raise

        tool_name = getattr(self.metadata, "name", None)
        if not tool_name and hasattr(self.metadata, "get_name"):
            tool_name = self.metadata.get_name()

        return CustomToolOutput(
            content=str(response),
            tool_name=tool_name or "query_engine_tool",
            raw_input={"input": input},
            raw_output=response,
        )

    async def acall(self, input: Any) -> CustomToolOutput:
        query_str = cast(str, input)
        logging.info(f"Starting asynchronous query engine tool Query with content: {query_str}")

        try:
            response = await self._query_engine.aquery(query_str)
            logging.info(f"Received asynchronous response from query engine tool: {response}")
        except Exception as e:
            logging.error(f"Asynchronous query from query engine tool failed: {e}", exc_info=True)
            raise

        tool_name = getattr(self.metadata, "name", None)
        if not tool_name and hasattr(self.metadata, "get_name"):
            tool_name = self.metadata.get_name()

        return CustomToolOutput(
            content=str(response),
            tool_name=tool_name or "query_engine_tool",
            raw_input={"input": input},
            raw_output=response,
        )
