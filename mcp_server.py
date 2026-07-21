# ============================================================================
# mcp_server.py — Spectral Memory Manifold Co-Processor
# MCP STDIO SERVER: JSON-RPC 2.0 message loop, tool registration
# Synchronous protocol. No asyncio. Thread-safe with RLock.
# ============================================================================

import json
import os
import signal
import sys
import threading
import time
import traceback
from typing import Any, Callable

import structlog
from pydantic import BaseModel, Field, ValidationError

from quantum_gate import QuantumGate, CollapseRequest, CollapseResult

logger = structlog.get_logger(__name__)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# JSON-RPC 2.0 Pydantic Models
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class JSONRPCErrorDetail(BaseModel):
    """JSON-RPC 2.0 error object."""

    code: int
    message: str
    data: Any = None


class JSONRPCRequest(BaseModel):
    """JSON-RPC 2.0 request object."""

    jsonrpc: str = Field(default="2.0")
    id: int | str | None = None
    method: str
    params: dict[str, Any] = Field(default_factory=dict)


class JSONRPCResponse(BaseModel):
    """JSON-RPC 2.0 success response object."""

    jsonrpc: str = Field(default="2.0")
    id: int | str | None = None
    result: Any = None


class JSONRPCError(BaseModel):
    """JSON-RPC 2.0 error response object."""

    jsonrpc: str = Field(default="2.0")
    id: int | str | None = None
    error: JSONRPCErrorDetail


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Error codes (JSON-RPC 2.0 spec + custom)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

ERROR_INVALID_REQUEST = -32600
ERROR_METHOD_NOT_FOUND = -32601
ERROR_INVALID_PARAMS = -32602
ERROR_INTERNAL = -32603
ERROR_MANIFOLD_NOT_INITIALIZED = -32000
ERROR_QUERY_EMBEDDING_FAILED = -32001
ERROR_EMPTY_RESULT_SET = -32002

ERROR_MESSAGES: dict[int, str] = {
    ERROR_INVALID_REQUEST: "Invalid Request",
    ERROR_METHOD_NOT_FOUND: "Method not found",
    ERROR_INVALID_PARAMS: "Invalid params",
    ERROR_INTERNAL: "Internal error",
    ERROR_MANIFOLD_NOT_INITIALIZED: "Manifold not initialized",
    ERROR_QUERY_EMBEDDING_FAILED: "Query embedding failed",
    ERROR_EMPTY_RESULT_SET: "Empty result set",
}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Tool Registration
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class ToolSchema(BaseModel):
    """Schema descriptor for a registered tool."""

    name: str
    description: str
    input_schema: dict[str, Any]


class ToolRegistration:
    """Holds a registered tool's handler and schema."""

    def __init__(
        self,
        name: str,
        handler: Callable[..., CollapseResult],
        schema: dict[str, Any],
        description: str = "",
    ) -> None:
        self.name = name
        self.handler = handler
        self.schema = schema
        self.description = description


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# MCP Server
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class MCPServer:
    """Model Context Protocol (MCP) server over stdio.

    Synchronous JSON-RPC 2.0 message loop. Thread-safe tool registry.
    Handles `tools/list` and `tools/call` MCP methods.
    """

    def __init__(self, quantum_gate: QuantumGate) -> None:
        """Initialise the MCP server with a QuantumGate instance.

        Args:
            quantum_gate: Initialised QuantumGate for collapsing queries.
        """
        self.quantum_gate = quantum_gate
        self._tools: dict[str, ToolRegistration] = {}
        self._running = threading.Event()
        self._lock = threading.RLock()

        # Signal handling for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        # Register default tools
        self._register_collapse_tool()

        logger.info("MCPServer initialized")

    def _signal_handler(self, signum: int, _frame: Any) -> None:
        """Handle SIGINT/SIGTERM for graceful shutdown.

        Args:
            signum: Signal number.
            _frame: Current stack frame (unused).
        """
        logger.info("received_shutdown_signal", signal=signum)
        self._running.clear()

    def register_tool(
        self,
        name: str,
        handler: Callable[..., CollapseResult],
        schema: dict[str, Any],
        description: str = "",
    ) -> None:
        """Register a tool with the MCP server.

        Args:
            name: Tool name (used in tools/call method).
            handler: Callable that takes keyword arguments and returns CollapseResult.
            schema: JSON Schema describing the tool's parameters.
            description: Human-readable description of the tool.
        """
        with self._lock:
            self._tools[name] = ToolRegistration(
                name=name, handler=handler, schema=schema, description=description
            )
            logger.debug("tool_registered", name=name)

    def _register_collapse_tool(self) -> None:
        """Register the collapse_quantum_memory tool."""
        collapse_schema = {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "The natural language query"},
                "max_tokens": {
                    "type": "integer",
                    "description": "Maximum tokens in response",
                    "default": 4096,
                    "minimum": 1,
                    "maximum": 128000,
                },
                "temperature": {
                    "type": "number",
                    "description": "Ignored — determinism enforced",
                    "default": 0.0,
                },
                "required_concepts": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Concepts that MUST appear in results",
                    "default": [],
                },
                "session_id": {
                    "type": "string",
                    "description": "Session identifier",
                    "default": "default",
                },
            },
            "required": ["query"],
        }

        self.register_tool(
            name="collapse_quantum_memory",
            handler=self.quantum_gate.collapse,
            schema=collapse_schema,
            description=(
                "Collapse a query into the spectral memory manifold. "
                "Returns retrieved memory pages ranked by submodular relevance. "
                "Deterministic output for identical inputs."
            ),
        )

    def _parse_request(self, raw_line: str) -> JSONRPCRequest | None:
        """Parse a raw JSON line into a JSONRPCRequest.

        Args:
            raw_line: Raw JSON string from stdin.

        Returns:
            Parsed JSONRPCRequest, or None if parsing failed.
        """
        try:
            data = json.loads(raw_line)
            return JSONRPCRequest(**data)
        except (json.JSONDecodeError, ValidationError) as exc:
            error_detail = JSONRPCErrorDetail(
                code=ERROR_INVALID_REQUEST,
                message=ERROR_MESSAGES[ERROR_INVALID_REQUEST],
                data={"parse_error": str(exc), "raw": raw_line[:200]},
            )
            error_response = JSONRPCError(
                id=None, error=error_detail
            )
            self._send_response(error_response.model_dump(exclude_none=True))
            return None

    def _handle_tools_list(self, request: JSONRPCRequest) -> None:
        """Handle the 'tools/list' MCP method.

        Args:
            request: The parsed JSON-RPC request.
        """
        with self._lock:
            tools_list = [
                {
                    "name": name,
                    "description": reg.description,
                    "inputSchema": reg.schema,
                }
                for name, reg in self._tools.items()
            ]

        response = JSONRPCResponse(
            id=request.id,
            result={"tools": tools_list},
        )
        self._send_response(response.model_dump(exclude_none=True))

    def _handle_tools_call(self, request: JSONRPCRequest) -> None:
        """Handle the 'tools/call' MCP method.

        Args:
            request: The parsed JSON-RPC request.
        """
        tool_name = request.params.get("name", "")
        arguments = request.params.get("arguments", {})

        if not tool_name or tool_name not in self._tools:
            error_detail = JSONRPCErrorDetail(
                code=ERROR_METHOD_NOT_FOUND,
                message=ERROR_MESSAGES[ERROR_METHOD_NOT_FOUND],
                data={"tool": tool_name},
            )
            self._send_response(
                JSONRPCError(id=request.id, error=error_detail).model_dump(
                    exclude_none=True
                )
            )
            return

        reg = self._tools[tool_name]

        try:
            # Validate arguments against CollapseRequest schema
            collapse_req = CollapseRequest(**arguments)

            # Execute the handler
            result = reg.handler(
                query=collapse_req.query,
                max_tokens=collapse_req.max_tokens,
                required_concepts=collapse_req.required_concepts,
            )

            # Serialise CollapseResult to dict for JSON-RPC response
            result_dict = result.model_dump(exclude_none=True)

            response = JSONRPCResponse(
                id=request.id,
                result={
                    "content": [
                        {
                            "type": "text",
                            "text": json.dumps(result_dict),
                        }
                    ],
                    "isError": result.error is not None,
                },
            )
            self._send_response(response.model_dump(exclude_none=True))

        except ValidationError as exc:
            error_detail = JSONRPCErrorDetail(
                code=ERROR_INVALID_PARAMS,
                message=ERROR_MESSAGES[ERROR_INVALID_PARAMS],
                data={"validation_error": exc.errors(), "arguments": arguments},
            )
            self._send_response(
                JSONRPCError(id=request.id, error=error_detail).model_dump(
                    exclude_none=True
                )
            )
        except Exception as exc:
            logger.exception("tool_call_failed", tool=tool_name, error=str(exc))
            error_detail = JSONRPCErrorDetail(
                code=ERROR_INTERNAL,
                message=ERROR_MESSAGES[ERROR_INTERNAL],
                data={
                    "tool": tool_name,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                },
            )
            self._send_response(
                JSONRPCError(id=request.id, error=error_detail).model_dump(
                    exclude_none=True
                )
            )

    def _handle_initialize(self, request: JSONRPCRequest) -> None:
        """Handle the 'initialize' MCP method.

        Args:
            request: The parsed JSON-RPC request.
        """
        response = JSONRPCResponse(
            id=request.id,
            result={
                "protocolVersion": "2024-11-05",
                "capabilities": {
                    "tools": {
                        "listChanged": False,
                    },
                    "resources": {},
                    "prompts": {},
                },
                "serverInfo": {
                    "name": "spectral-memory-manifold",
                    "version": "1.0.0",
                },
            },
        )
        self._send_response(response.model_dump(exclude_none=True))

    def _send_response(self, data: dict[str, Any]) -> None:
        """Send a JSON-RPC response to stdout.

        Args:
            data: Dictionary to serialise as JSON and write to stdout.
        """
        try:
            line = json.dumps(data) + "\n"
            sys.stdout.write(line)
            sys.stdout.flush()
        except OSError as exc:
            logger.error("send_response_failed", error=str(exc))

    def _dispatch(self, request: JSONRPCRequest) -> None:
        """Dispatch a parsed request to the appropriate handler.

        Args:
            request: Parsed JSON-RPC request.
        """
        method = request.method

        try:
            if method == "initialize":
                self._handle_initialize(request)
            elif method == "tools/list":
                self._handle_tools_list(request)
            elif method == "tools/call":
                self._handle_tools_call(request)
            elif method == "notifications/initialized":
                # Notification — no response needed
                logger.debug("client_initialized")
                pass
            else:
                error_detail = JSONRPCErrorDetail(
                    code=ERROR_METHOD_NOT_FOUND,
                    message=ERROR_MESSAGES[ERROR_METHOD_NOT_FOUND],
                    data={"method": method},
                )
                self._send_response(
                    JSONRPCError(id=request.id, error=error_detail).model_dump(
                        exclude_none=True
                    )
                )
        except Exception as exc:
            logger.exception("dispatch_failed", method=method, error=str(exc))
            error_detail = JSONRPCErrorDetail(
                code=ERROR_INTERNAL,
                message=ERROR_MESSAGES[ERROR_INTERNAL],
                data={"method": method, "error": str(exc)},
            )
            self._send_response(
                JSONRPCError(id=request.id, error=error_detail).model_dump(
                    exclude_none=True
                )
            )

    def run(self) -> None:
        """Run the MCP server message loop.

        Reads JSON-RPC messages line-by-line from stdin, dispatches them,
        and writes responses to stdout. Blocks until stdin is closed or
        SIGINT/SIGTERM is received.
        """
        self._running.set()
        logger.info("MCP server started")

        while self._running.is_set():
            try:
                raw_line = sys.stdin.readline()
                if not raw_line:
                    # EOF on stdin
                    logger.info("stdin_closed")
                    break

                raw_line = raw_line.strip()
                if not raw_line:
                    continue

                request = self._parse_request(raw_line)
                if request is not None:
                    self._dispatch(request)

            except KeyboardInterrupt:
                logger.info("keyboard_interrupt")
                break
            except Exception as exc:
                logger.error("message_loop_error", error=str(exc))
                # Attempt to send an error response
                error_detail = JSONRPCErrorDetail(
                    code=ERROR_INTERNAL,
                    message=ERROR_MESSAGES[ERROR_INTERNAL],
                    data={"error": str(exc), "traceback": traceback.format_exc()},
                )
                self._send_response(
                    JSONRPCError(id=None, error=error_detail).model_dump(
                        exclude_none=True
                    )
                )

        logger.info("MCP server stopped")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Factory function
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def create_server(
    quantum_gate: QuantumGate,
) -> MCPServer:
    """Create and return a configured MCP server instance.

    Args:
        quantum_gate: Initialised QuantumGate instance.

    Returns:
        Configured MCPServer ready to run.
    """
    return MCPServer(quantum_gate=quantum_gate)
