"""
Document Oracle for DSRO (Document Search as Repair Obligation)

This module provides a simple, non-intelligent document lookup service.
It performs NO reasoning, NO summarization, NO semantic understanding.

The oracle:
- Parses documents into sections with hierarchical IDs
- Builds a keyword index for string matching
- Returns raw text for probes

Design invariant:
"The oracle performs no reasoning."
"""

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple
from pathlib import Path


@dataclass
class Section:
    """A document section with hierarchical ID."""
    id: str                    # e.g., "rules/single-value"
    title: str                 # e.g., "Single-Value Rules"
    content: str               # Raw text content
    line_start: int            # Starting line number
    line_end: int              # Ending line number
    level: int                 # Hierarchy depth (0 = top-level)
    parent_id: Optional[str]   # Parent section ID


@dataclass
class DocumentObservation:
    """Result of a document probe."""
    probe_kind: str            # "open_section" or "search_keyword"
    probe_target: str          # Section ID or keyword
    found: bool                # Whether probe found anything
    text: str                  # Retrieved text (empty if not found)
    section_id: Optional[str]  # Section ID where found (for keyword search)

    def to_dict(self) -> dict:
        return {
            "probe": {
                "kind": self.probe_kind,
                "target": self.probe_target,
            },
            "found": self.found,
            "text": self.text,
            "section_id": self.section_id,
        }


@dataclass
class DocumentIndex:
    """Indexed document ready for probing."""
    source_path: str
    sections: Dict[str, Section] = field(default_factory=dict)
    keyword_index: Dict[str, List[str]] = field(default_factory=dict)  # keyword -> [section_ids]
    all_section_ids: List[str] = field(default_factory=list)

    def get_section_ids(self) -> List[str]:
        """Return all available section IDs."""
        return self.all_section_ids.copy()

    def get_keywords(self) -> List[str]:
        """Return all indexed keywords."""
        return list(self.keyword_index.keys())


class DocumentParser:
    """
    Parses plain text documents into sections.

    Detects sections based on formatting patterns:
    - Lines ending with specific patterns (e.g., "Rules", "—")
    - Indentation/hierarchy based on markers
    """

    # Patterns for detecting section headers
    HEADER_PATTERNS = [
        # "Title — Description" style headers
        r'^([A-Z][A-Za-z\s]+)\s*[—\-]+\s*.+$',
        # Simple capitalized headers (e.g., "Some", "Every")
        r'^([A-Z][a-z]+)$',
        # "Objects — Extracting Data" style
        r'^(Objects\s*[—\-]+\s*\w+.*)$',
    ]

    def __init__(self):
        self.header_re = [re.compile(p) for p in self.HEADER_PATTERNS]

    def parse(self, filepath: str) -> DocumentIndex:
        """Parse a document file into an indexed structure."""
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"Document not found: {filepath}")

        content = path.read_text()
        lines = content.split('\n')

        index = DocumentIndex(source_path=filepath)

        # First pass: identify all headers and their positions
        headers = self._find_headers(lines)

        # Second pass: build sections with content
        sections = self._build_sections(lines, headers)

        # Build section dictionary and ID list
        for section in sections:
            index.sections[section.id] = section
            index.all_section_ids.append(section.id)

        # Build keyword index
        index.keyword_index = self._build_keyword_index(sections)

        return index

    def _find_headers(self, lines: List[str]) -> List[Tuple[int, str, int]]:
        """
        Find all headers in the document.
        Returns: [(line_number, title, level), ...]
        """
        headers = []

        for i, line in enumerate(lines):
            stripped = line.strip()
            if not stripped:
                continue

            # Check for main section headers (with em-dash)
            if '—' in stripped or ' - ' in stripped:
                # Main section header
                title = stripped.split('—')[0].split(' - ')[0].strip()
                if title and title[0].isupper():
                    headers.append((i, stripped, 0))
                    continue

            # Check for subsection headers (single capitalized word followed by content)
            # These are typically like "Some", "Every", "Print", etc.
            if (stripped and
                stripped[0].isupper() and
                len(stripped) < 50 and
                not stripped.endswith(('.', ',', ':', ';')) and
                i + 1 < len(lines) and
                lines[i + 1].strip()):

                # Check if next line looks like description (starts lowercase or is code)
                next_line = lines[i + 1].strip()
                if (next_line and
                    (next_line[0].islower() or
                     next_line.startswith(('Name ', 'Check ', 'Use ', 'Override ', 'Produce ', 'Pattern ', 'Summarize ')))):
                    headers.append((i, stripped, 1))

        return headers

    def _build_sections(self, lines: List[str], headers: List[Tuple[int, str, int]]) -> List[Section]:
        """Build section objects with content."""
        sections = []

        if not headers:
            # No headers found, treat entire document as one section
            return [Section(
                id="document",
                title="Document",
                content='\n'.join(lines),
                line_start=0,
                line_end=len(lines) - 1,
                level=0,
                parent_id=None,
            )]

        # Process each header
        current_parent_id = None
        current_parent_level = -1

        for i, (line_num, title, level) in enumerate(headers):
            # Determine end line (start of next header or end of document)
            if i + 1 < len(headers):
                end_line = headers[i + 1][0] - 1
            else:
                end_line = len(lines) - 1

            # Skip empty trailing lines
            while end_line > line_num and not lines[end_line].strip():
                end_line -= 1

            # Extract content (excluding the header line itself)
            content_lines = lines[line_num:end_line + 1]
            content = '\n'.join(content_lines)

            # Generate section ID
            section_id = self._generate_section_id(title)

            # Determine parent
            if level == 0:
                current_parent_id = section_id
                current_parent_level = 0
                parent_id = None
            else:
                parent_id = current_parent_id
                section_id = f"{current_parent_id}/{section_id}" if current_parent_id else section_id

            sections.append(Section(
                id=section_id,
                title=title,
                content=content,
                line_start=line_num,
                line_end=end_line,
                level=level,
                parent_id=parent_id,
            ))

        return sections

    def _generate_section_id(self, title: str) -> str:
        """Generate a URL-safe section ID from title."""
        # For titles with em-dash, check if main part needs disambiguation
        if '—' in title:
            parts = title.split('—')
            main = parts[0].strip().lower()
            desc = parts[1].strip().lower() if len(parts) >= 2 else ""

            # Special case: "Objects" appears multiple times, needs disambiguation
            if main == "objects" and desc:
                # Use first word of descriptor
                desc_word = re.sub(r'[^a-z0-9]', '', desc.split()[0]) if desc.split() else ""
                section_id = f"{main}-{desc_word}"
            else:
                section_id = main
        else:
            section_id = title.split(' - ')[0].strip().lower()

        # Clean up: keep only alphanumeric, spaces, and hyphens, then convert spaces to hyphens
        section_id = re.sub(r'[^a-z0-9\s\-]', '', section_id)
        section_id = re.sub(r'\s+', '-', section_id.strip())

        return section_id

    def _build_keyword_index(self, sections: List[Section]) -> Dict[str, List[str]]:
        """
        Build keyword index from sections.

        Indexes:
        - Rego keywords (if, contains, some, every, etc.)
        - Built-in function names
        - Common terms
        """
        # Keywords to index
        REGO_KEYWORDS = {
            'if', 'contains', 'some', 'every', 'in', 'with', 'as',
            'default', 'true', 'false', 'null', 'not',
        }

        BUILTIN_FUNCTIONS = {
            'print', 'sprintf', 'concat', 'count', 'sum', 'max', 'min', 'sort',
            'startswith', 'endswith', 'contains', 'replace', 'split', 'trim',
            'regex.match', 'regex.replace', 'regex.find_all_string_submatch',
            'object.get', 'object.keys', 'object.union', 'object.subset', 'object.remove',
            'array.concat', 'array.slice',
            'json.marshal', 'json.unmarshal',
            'io.jwt.decode', 'io.jwt.verify',
        }

        index: Dict[str, List[str]] = {}

        for section in sections:
            content_lower = section.content.lower()

            # Index Rego keywords
            for kw in REGO_KEYWORDS:
                if kw in content_lower:
                    if kw not in index:
                        index[kw] = []
                    if section.id not in index[kw]:
                        index[kw].append(section.id)

            # Index builtin functions
            for func in BUILTIN_FUNCTIONS:
                if func in section.content:  # Case-sensitive for functions
                    if func not in index:
                        index[func] = []
                    if section.id not in index[func]:
                        index[func].append(section.id)

            # Index words from section title
            title_words = re.findall(r'\b[a-z]{3,}\b', section.title.lower())
            for word in title_words:
                if word not in index:
                    index[word] = []
                if section.id not in index[word]:
                    index[word].append(section.id)

        return index


class DocumentOracle:
    """
    The Document Oracle - a simple lookup service.

    Provides two probe types:
    - open_section: Retrieve a section by ID
    - search_keyword: Find sections containing a keyword

    The oracle performs NO reasoning.
    """

    def __init__(self, index: DocumentIndex):
        self.index = index
        self.probe_history: List[Tuple[str, str]] = []  # [(kind, target), ...]

    def reset(self) -> None:
        """Reset probe history for a new episode."""
        self.probe_history = []

    def get_available_sections(self) -> List[str]:
        """Return list of all available section IDs."""
        return self.index.get_section_ids()

    def get_available_keywords(self) -> List[str]:
        """Return list of all indexed keywords."""
        return self.index.get_keywords()

    def is_probe_repeated(self, kind: str, target: str) -> bool:
        """Check if this exact probe has been made before."""
        return (kind, target) in self.probe_history

    def get_probe_count(self) -> int:
        """Return number of probes made."""
        return len(self.probe_history)

    def get_remaining_sections(self) -> List[str]:
        """Return section IDs that haven't been probed yet."""
        probed_sections = {t for k, t in self.probe_history if k == "open_section"}
        return [s for s in self.index.all_section_ids if s not in probed_sections]

    def get_remaining_keywords(self) -> List[str]:
        """Return keywords that haven't been probed yet."""
        probed_keywords = {t for k, t in self.probe_history if k == "search_keyword"}
        return [k for k in self.index.keyword_index.keys() if k not in probed_keywords]

    def is_exhausted(self) -> bool:
        """Check if all possible probes have been made."""
        return (len(self.get_remaining_sections()) == 0 and
                len(self.get_remaining_keywords()) == 0)

    def probe(self, kind: str, target: str) -> DocumentObservation:
        """
        Perform a document probe.

        Args:
            kind: "open_section" or "search_keyword"
            target: Section ID or keyword to search

        Returns:
            DocumentObservation with retrieved text

        Note: Does NOT enforce non-repetition. That's the supervisor's job.
        """
        # Record probe in history
        self.probe_history.append((kind, target))

        if kind == "open_section":
            return self._probe_section(target)
        elif kind == "search_keyword":
            return self._probe_keyword(target)
        else:
            return DocumentObservation(
                probe_kind=kind,
                probe_target=target,
                found=False,
                text=f"Unknown probe kind: {kind}",
                section_id=None,
            )

    def _probe_section(self, section_id: str) -> DocumentObservation:
        """Retrieve a section by ID."""
        section = self.index.sections.get(section_id)

        if section is None:
            return DocumentObservation(
                probe_kind="open_section",
                probe_target=section_id,
                found=False,
                text=f"Section not found: {section_id}",
                section_id=None,
            )

        return DocumentObservation(
            probe_kind="open_section",
            probe_target=section_id,
            found=True,
            text=section.content,
            section_id=section_id,
        )

    def _probe_keyword(self, keyword: str) -> DocumentObservation:
        """Search for sections containing a keyword."""
        # First check the keyword index
        section_ids = self.index.keyword_index.get(keyword, [])

        if not section_ids:
            # Try case-insensitive search across all sections
            keyword_lower = keyword.lower()
            for section in self.index.sections.values():
                if keyword_lower in section.content.lower():
                    section_ids.append(section.id)

        if not section_ids:
            return DocumentObservation(
                probe_kind="search_keyword",
                probe_target=keyword,
                found=False,
                text=f"No sections found containing: {keyword}",
                section_id=None,
            )

        # Return content from all matching sections
        texts = []
        for sid in section_ids:
            section = self.index.sections[sid]
            texts.append(f"=== {section.title} ({sid}) ===\n{section.content}")

        return DocumentObservation(
            probe_kind="search_keyword",
            probe_target=keyword,
            found=True,
            text="\n\n".join(texts),
            section_id=section_ids[0],  # Primary section
        )


def load_document(filepath: str) -> DocumentOracle:
    """
    Convenience function to load and index a document.

    Usage:
        oracle = load_document("agent-docs/rego_cheat_sheet.txt")
        result = oracle.probe("open_section", "rules")
    """
    parser = DocumentParser()
    index = parser.parse(filepath)
    return DocumentOracle(index)
