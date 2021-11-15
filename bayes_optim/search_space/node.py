from __future__ import annotations

from copy import copy, deepcopy
from typing import Any, Dict, List, Tuple


class Node:
    r"""Node of a tree used to represent the conditional search space"""

    def __init__(self, name: str, data: Any = None):
        """create a node in a tree structure

        Parameters
        ----------
        name : str
            name of the node
        data : Any, optional
            any data stored in this node, by default None
        """
        self.name: str = name
        self.data: Any = data
        self.is_root: bool = True  # is this node a root?
        self.children: List[Node] = []
        self.branches: List = []  # branching conditions

    def add_child(self, node: Node, branch: str = None) -> Node:
        """add a child node to a node

        Parameters
        ----------
        node : Node
            the child node to add
        branch : str, optional
            the branching condition to this child, which is an Python expression evaluated to
            `True` when the conditiion is satisfied, by default None

        Returns
        -------
        Node
            return the updated node
        """
        node.is_root = False
        self.children.append(node)
        self.branches.append(branch)
        return self

    def add_child_from_dict(self, d: dict, cache: dict = None) -> Node:
        """enumerate a dictionary describing a tree structure and add children nodes to the current
        one recursively.

        Parameters
        ----------
        d : dict
            The input dictionary describing the tree structure. See below for an example.
        cache : dict, optional
            intermediate cache used in the recursion, by default None

        Returns
        -------
        Node
            the updated current node with all its children nodes added from the input dictionary.
        """
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

    def deepcopy(self) -> Node:
        """copy the whole tree recursively, starting from the root node

        Returns
        -------
        Node
            the root node of the copied tree
        """
        return self.remove([])

    def remove(self, node_names: List[str], invert: bool = False) -> Node:
        """remove the node listed in `node_names` where the entire sub-tree is dropped if its
        root node is removed. The resulting node is a copy of the original one.

        Parameters
        ----------
        node_names : List[str]
            a list of name to drop/keep
        invert : bool, optional
            whether to drop `node_names` (`invert = False`) or keep (`invert = True`),
            by default False

        Returns
        -------
        Node
            the updated root node
        """
        op = lambda e, S: e not in S if invert else lambda e, S: e in S
        if op(self.name, node_names):
            return None
        node = Node(self.name, deepcopy(self.data))
        if self.children:  # an internal node
            for i, child in enumerate(self.children):
                if not op(child.name, node_names):
                    node.add_child(Node.remove(child, node_names, invert), copy(self.branches[i]))
        return node

    @classmethod
    def from_dict(cls, d: dict) -> List[Node]:
        """create trees recursively from a dictionary

        Parameters
        ----------
        d : dict
            The input dictionary describing the tree structure. See below for an example.

        Returns
        -------
        List[Node]
            a list of root nodes as the user can specify multiple trees in the input.
        """
        cache = dict()
        for k in d.keys():
            if k in cache:
                continue
            cls(k).add_child_from_dict(d, cache)
        return [node for _, node in cache.items() if node.is_root]

    def to_dict(self) -> dict:
        out = dict()
        if self.children:  # an internal node
            for i, child in enumerate(self.children):
                out.setdefault(self.name, []).append(
                    {"name": child.name, "condition": self.branches[i]}
                )
                out.update(child.to_dict())
        return out

    def pprint(self, _prefix: str = "", branch: str = None, _last: bool = True, data: dict = None):
        s_branch = "`- " if _last else "|- "
        s_branch += f"<{branch}> - " if branch else ""
        print(
            _prefix,
            s_branch,
            data[self.name] if data else self.name,
            sep="",
        )
        if not _last:
            _prefix += "|"
        _prefix += " " * len(s_branch)
        child_count = len(self.children)
        for i, child in enumerate(self.children):
            _last = i == (child_count - 1)
            Node.pprint(child, _prefix, self.branches[i], _last, data)

    def get_all_name(self) -> List[str]:
        """Traverse the tree in pre-order and extract the node's names"""
        out = [self.name]
        if self.children:
            for child in self.children:
                out += Node.get_all_name(child)
        return out

    def get_all_path(self) -> Dict[Tuple[str], List[str]]:
        """Traverse the tree in pre-order get all path to leaf nodes"""
        if not self.children:
            return {(): None}

        path = dict()
        for br in set(self.branches):
            _path = [
                ((br,) + p, v) if p else ((br,), [self.children[i].name])
                for i in [i for i, b in enumerate(self.branches) if b == br]
                for p, v in Node.get_all_path(self.children[i]).items()
            ]
            _merged = dict()
            for k, v in _path:
                for vv in v:
                    _merged.setdefault(k, []).append(vv)
            path.update(_merged)
        return path

    def __str__(self) -> str:
        return self.name

    def __repr__(self) -> str:
        return self.name
