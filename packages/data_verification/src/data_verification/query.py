"""query: run a read-only SPARQL query over a store (local rdflib, no server)."""

from __future__ import annotations

import csv
import io
import json

from rdflib import Graph, Literal, URIRef

from .graph import CP
from .store import Store

PREFIXES = {
    "cp": str(CP),
    "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
    "rdfs": "http://www.w3.org/2000/01/rdf-schema#",
    "owl": "http://www.w3.org/2002/07/owl#",
    "xsd": "http://www.w3.org/2001/XMLSchema#",
}


def query(store: Store, text: str, fmt: str = "table", with_ontology: bool = False) -> str:
    """SELECT -> table, csv or json; ASK -> true/false; CONSTRUCT/DESCRIBE -> Turtle."""
    graph = store.graph(with_ontology=with_ontology)
    graph.bind("ds", f"{store.dataset}/")
    try:
        result = graph.query(text, initNs=PREFIXES)
    except Exception as exc:  # rdflib raises parser and evaluation errors of several types
        raise ValueError(f"invalid or unsupported SPARQL query: {exc}") from None
    if result.type == "ASK":
        return "true" if result.askAnswer else "false"
    if result.type in ("CONSTRUCT", "DESCRIBE"):
        out = Graph()
        for prefix, ns in graph.namespaces():
            out.bind(prefix, ns)
        for triple in result:
            out.add(triple)
        return out.serialize(format="turtle")
    names = [str(v) for v in result.vars]
    rows = [[_cell(row[v], graph) for v in result.vars] for row in result]
    if fmt == "json":
        return json.dumps([dict(zip(names, (_json(row[v]) for v in result.vars))) for row in result], indent=2)
    if fmt == "csv":
        buffer = io.StringIO()
        writer = csv.writer(buffer)
        writer.writerow(names)
        writer.writerows(rows)
        return buffer.getvalue().rstrip("\n")
    return _table(names, rows)


def _cell(term, graph: Graph) -> str:
    if term is None:
        return ""
    if isinstance(term, URIRef):
        return term.n3(graph.namespace_manager)
    return str(term)


def _json(term):
    if term is None:
        return None
    if isinstance(term, Literal):
        value = term.toPython()
        return value if isinstance(value, (int, float, bool, str)) else str(value)
    return str(term)


def _table(names: list[str], rows: list[list[str]]) -> str:
    widths = [max(len(n), *(len(r[i]) for r in rows)) if rows else len(n) for i, n in enumerate(names)]

    def line(cells: list[str]) -> str:
        return "  ".join(c.ljust(w) for c, w in zip(cells, widths)).rstrip()

    out = [line(names), line(["-" * w for w in widths])] + [line(r) for r in rows]
    out.append(f"({len(rows)} row{'s' if len(rows) != 1 else ''})")
    return "\n".join(out)
