"""Read and write Newick.

Iterative, because these trees have >100,000 leaves and a recursive parser overflows the stack.
Quoted labels and `[...]` comments are handled: TreeTime writes NEXUS-flavoured Newick with
comments, and quoted labels appear wherever a strain name contains a bracket or a comma.
"""

from __future__ import annotations

from pathlib import Path

from af.tree.model import Node, Tree, TreeError

_LABEL_END = "(),:[ \t\r\n"


def loads(text: str) -> Tree:
    """Parse one Newick tree. Ids are not assigned; call ``Tree.assign_ids()`` when ready."""
    text = text.strip()
    if not text:
        raise TreeError("empty Newick input")
    semicolon = text.find(";")
    if semicolon != -1:
        text = text[:semicolon]
    root = Node()
    current = root
    stack: list[Node] = []
    index, end = 0, len(text)
    while index < end:
        char = text[index]
        if char == "(":
            child = Node(parent=current)
            current.children.append(child)
            stack.append(current)
            current = child
            index += 1
        elif char == ",":
            if not stack:
                raise TreeError("',' outside any parentheses")
            parent = stack[-1]
            current = Node(parent=parent)
            parent.children.append(current)
            index += 1
        elif char == ")":
            if not stack:
                raise TreeError("unbalanced ')' in Newick")
            current = stack.pop()
            index += 1
        elif char in " \t\r\n":
            index += 1
        elif char == "[":
            close = text.find("]", index)
            if close == -1:
                raise TreeError("unterminated '[' comment in Newick")
            index = close + 1
        elif char == ":":
            index += 1
            start = index
            while index < end and text[index] not in _LABEL_END:
                index += 1
            try:
                current.branch_length = float(text[start:index])
            except ValueError as error:
                raise TreeError(f"bad branch length {text[start:index]!r}") from error
        else:
            current.name, index = _read_label(text, index)
    if stack:
        raise TreeError(f"unbalanced '(' in Newick: {len(stack)} never closed")
    tree = Tree(root)
    for node in tree.preorder():
        if node.is_leaf and not node.name:
            raise TreeError("a leaf has no label; af addresses leaves by name, never by position")
    return tree


def _read_label(text: str, index: int) -> tuple[str, int]:
    if text[index] == "'":
        out: list[str] = []
        index += 1
        while index < len(text):
            if text[index] == "'":
                if index + 1 < len(text) and text[index + 1] == "'":
                    out.append("'")
                    index += 2
                    continue
                return "".join(out), index + 1
            out.append(text[index])
            index += 1
        raise TreeError("unterminated quoted label in Newick")
    start = index
    while index < len(text) and text[index] not in _LABEL_END:
        index += 1
    return text[start:index], index


def load(path: Path) -> Tree:
    return loads(Path(path).read_text())


def dumps(tree: Tree, with_internal_labels: bool = False, precision: int = 10) -> str:
    """Write Newick, optionally labelling internal nodes with their stable ids.

    Writing the ids means a tool that preserves internal labels (raxml-ng, IQ-TREE) lets ancestral
    states be matched back by identity rather than by position, which is what makes the ASR
    backends interchangeable.
    """
    if with_internal_labels and not tree.ids_assigned():
        raise TreeError("ids have not been assigned; call assign_ids() before writing them")
    out: list[str] = []
    # Stack of either a node to emit, or a literal string to append.
    stack: list[Node | str] = [tree.root]
    while stack:
        item = stack.pop()
        if isinstance(item, str):
            out.append(item)
            continue
        suffix = "" if item.parent is None else f":{item.branch_length:.{precision}g}"
        if item.is_leaf:
            out.append(_quote(item.name or "") + suffix)
            continue
        label = item.id_hex if with_internal_labels else (item.name or "")
        out.append("(")
        stack.append(")" + _quote(label) + suffix if label else ")" + suffix)
        for index, child in enumerate(reversed(item.children)):
            if index:
                stack.append(",")
            stack.append(child)
    return "".join(out) + ";\n"


def _quote(label: str) -> str:
    if any(char in label for char in "(),:;[]' \t"):
        return "'" + label.replace("'", "''") + "'"
    return label


def dump(tree: Tree, path: Path, with_internal_labels: bool = False, precision: int = 10) -> None:
    Path(path).write_text(
        dumps(tree, with_internal_labels=with_internal_labels, precision=precision)
    )
