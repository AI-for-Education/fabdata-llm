## Anthropic `format_output` note

The combined change in `d83b831` and `f482ed5` appears directionally correct for Anthropic's documented extended-thinking responses, especially:

- `thinking -> text -> tool_use`
- `thinking -> tool_use`
- multiple `tool_use` blocks in one response

The main remaining concern is not a clearly documented live bug, but a potential parser assumption mismatch if Anthropic expands the range of content block orderings in the future.

The important regression to guard against is simpler and already known: after `d83b831`, tool-first responses with multiple `tool_use` blocks could lose the first tool call because later post-processing rebuilt `ToolCalls` from `content[1:]`. `f482ed5` fixes that by returning early from the tool-first branch.

Tests should cover the documented extended-thinking shapes above plus the multi-tool regression.
