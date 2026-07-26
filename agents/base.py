from dataclasses import dataclass, field
from typing import Any


@dataclass
class ResearchContext:
    query: str
    mode: str = "topic"
    tone: str = "default"
    sources: list = field(default_factory=list)
    report: str = ""
    followups: list = field(default_factory=list)
    contested: str = ""
    source_urls: set = field(default_factory=set)
    state: dict = field(default_factory=dict)
    error: str | None = None


class Agent:
    def __init__(self, name: str):
        self.name = name

    async def run(self, ctx: ResearchContext) -> ResearchContext:
        raise NotImplementedError
