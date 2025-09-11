from mcp.server.fastmcp import FastMCP

mcp = FastMCP("Calculator")


@mcp.tool()
def calculate_sum(a: float, b: float) -> float:
    """Calculate the sum of two numbers."""
    return a + b


if __name__ == "__main__":
    mcp.run()
