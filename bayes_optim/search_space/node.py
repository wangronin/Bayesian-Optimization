from __future__ import annotations

from typing import Any, List

# from py_expression_eval import Parser


class Node:
    r"""Node of a tree used to represent the conditional search space"""

    def __init__(self, name: str, data: Any = None):
        self.name: str = name
        self.data: Any = data
        self.is_root: bool = True  # is this node a root?
        self.children: List[Node] = []
        self.branches: List = []  # branching conditions

    def add_child(self, node: Node, branch: str = None) -> Node:
        node.is_root = False
        self.children.append(node)
        # TODO: add parsing exceptions here
        # expr = Parser().parse(branch) if branch else None
        self.branches.append(branch)
        return self

    def add_child_from_dict(self, d: dict, cache: dict = None) -> Node:
        if cache is None:
            cache = dict()
        children = d.get(self.name, None)
        if children:  # an internal node
            for info in children:
                key, branch = info["name"], info["condition"]
                self.add_child(
                    cache[key] if key in cache else Node.add_child_from_dict(Node(key), d, cache),
                    branch,
                )
            cache[self.name] = self
        return self

    @classmethod
    def from_dict(cls, d: dict) -> List[Node]:
        cache = dict()
        for k in d.keys():
            if k in cache:
                continue
            cls(k).add_child_from_dict(d, cache)
        return [node for _, node in cache.items() if node.is_root]

    def pprint(self, _prefix="", branch=None, _last=True):
        print(
            _prefix,
            "`- " if _last else "|- ",
            f"<{branch}> - " if branch else "",
            self.name,
            sep="",
        )
        _prefix += "   " if _last else "|  "
        child_count = len(self.children)
        for i, child in enumerate(self.children):
            _last = i == (child_count - 1)
            Node.pprint(child, _prefix, self.branches[i], _last)

    def eval_path(self, path: List) -> List[Node]:
        pass

    # def get_all_path(self) -> List[List]:
    #     if not self.children:
    #         return [[]]

    #     path = []
    #     # idx = {}
    #     # `set` for getting unique branches
    #     # d = {name: i for i, name in enumerate(set(self.branches))}
    #     # for key in self.branches:
    #     #     idx.setdefault(key, []).append(d[key])

    #     for br in set(self.branches):
    #         idx = [i for i, b in enumerate(self.branches) if b == br]
    #         _ = [Node.get_all_path(self.children[i]) for i in idx]
    #         _path = sum(_, [[]])
    #         breakpoint()
    #         path += [[br].append(p) if p is not None else [br] for p in _path]

    #     return path

    def __str__(self) -> str:
        return self.name

    def __repr__(self) -> str:
        return self.name
