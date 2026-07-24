"""Agent task-completion eval harness (Phase E).

A small benchmark of operator-goal scenarios (goal -> expected end-state) that
measures the agent surface as *tasks*, not units. Fully mocked by default
(0 real tokens, CI-safe); a live variant is gated behind ``NPA_AGENT_CHAT_LIVE=1``.
"""
