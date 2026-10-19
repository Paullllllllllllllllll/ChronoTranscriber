"""Deterministic EPUB and MOBI inputs for the characterization tests.

``make_epub`` writes a three-chapter EPUB through EbookLib. ``make_mobi``
writes a minimal Mobipocket 6 file by hand: a Palm database with one
uncompressed text record of HTML, a MOBI header, and an EXTH block that
names the author. The ``mobi`` library unpacks it to ``mobi7/book.html``.
"""

from __future__ import annotations

import struct
from collections.abc import Sequence
from pathlib import Path

from ebooklib import epub

BOOK_TITLE = "Annals of the Grain Market"
BOOK_AUTHOR = "Jean Characterization"

CHAPTERS: tuple[tuple[str, str], ...] = (
    (
        "Chapter One",
        "The magistrates fixed the bread assize in the spring of 1601.",
    ),
    (
        "Chapter Two",
        "Rye fell to nine sous after the harvest of 1602.",
    ),
    (
        "Chapter Three",
        "Wheat deliveries resumed along the river in 1603.",
    ),
)
"""(heading, body) of each chapter, in reading order."""

MOBI_TITLE = "Kindle Market Notes"
MOBI_AUTHOR = "Marie Fixture"
MOBI_PARAGRAPHS: tuple[str, ...] = (
    "Saffron arrived from Venice in 1610.",
    "Pepper prices doubled during the siege of 1611.",
)

_RECORD_SIZE = 4096
_MOBI_HEADER_LENGTH = 0xE8
_NONE = 0xFFFFFFFF


def chapter_html(heading: str, body: str) -> str:
    """Return the XHTML body of one chapter."""
    return f"<h1>{heading}</h1><p>{body}</p>"


def make_epub(
    path: Path,
    chapters: Sequence[tuple[str, str]] = CHAPTERS,
    *,
    title: str = BOOK_TITLE,
    author: str = BOOK_AUTHOR,
) -> Path:
    """Write an EPUB with one spine document per chapter, plus nav and NCX."""
    book = epub.EpubBook()
    book.set_identifier("ct-characterization-epub")
    book.set_title(title)
    book.set_language("en")
    book.add_author(author)
    items = []
    for index, (heading, body) in enumerate(chapters, 1):
        item = epub.EpubHtml(
            title=heading, file_name=f"chapter_{index:02d}.xhtml", lang="en"
        )
        item.content = chapter_html(heading, body)
        book.add_item(item)
        items.append(item)
    book.toc = items
    book.add_item(epub.EpubNcx())
    book.add_item(epub.EpubNav())
    book.spine = ["nav", *items]
    path.parent.mkdir(parents=True, exist_ok=True)
    epub.write_epub(str(path), book)
    return path


def mobi_html(
    title: str = MOBI_TITLE, paragraphs: Sequence[str] = MOBI_PARAGRAPHS
) -> str:
    """Return the HTML text record of the minimal MOBI."""
    body = "".join(f"<p>{text}</p>" for text in paragraphs)
    return (
        f"<html><head><title>{title}</title></head>"
        f"<body><h1>{title}</h1>{body}</body></html>"
    )


def _exth(author: str) -> bytes:
    value = author.encode("utf-8")
    record = struct.pack(">LL", 100, 8 + len(value)) + value
    length = 12 + len(record)
    padding = (-length) % 4
    return b"EXTH" + struct.pack(">LL", length, 1) + record + b"\0" * padding


def _record0(text_length: int, text_records: int, title: str, author: str) -> bytes:
    header = bytearray(16 + _MOBI_HEADER_LENGTH)
    struct.pack_into(
        ">HHLHHHH", header, 0, 1, 0, text_length, text_records, _RECORD_SIZE, 0, 0
    )
    header[0x10:0x14] = b"MOBI"
    struct.pack_into(">LLLLL", header, 0x14, _MOBI_HEADER_LENGTH, 2, 65001, 1, 6)
    for offset in range(0x28, 0x50, 4):
        struct.pack_into(">L", header, offset, _NONE)
    exth = _exth(author)
    name = title.encode("utf-8")
    name_offset = len(header) + len(exth)
    struct.pack_into(">LLL", header, 0x50, text_records + 1, name_offset, len(name))
    struct.pack_into(">L", header, 0x5C, 9)
    struct.pack_into(">L", header, 0x68, 6)
    struct.pack_into(">L", header, 0x6C, _NONE)
    struct.pack_into(">L", header, 0x80, 0x40)
    struct.pack_into(">L", header, 0xA4, _NONE)
    struct.pack_into(">L", header, 0xA8, _NONE)
    struct.pack_into(">HH", header, 0xC0, 1, text_records)
    struct.pack_into(">L", header, 0xC4, 1)
    for offset in (0xC8, 0xD0, 0xE0, 0xE8, 0xEC, 0xF4):
        struct.pack_into(">L", header, offset, _NONE)
    record = bytes(header) + exth + name
    return record + b"\0" * ((-len(record)) % 4 + 4)


def make_mobi(
    path: Path,
    *,
    title: str = MOBI_TITLE,
    author: str = MOBI_AUTHOR,
    paragraphs: Sequence[str] = MOBI_PARAGRAPHS,
) -> Path:
    """Write a minimal uncompressed Mobipocket 6 book holding ``mobi_html``."""
    text = mobi_html(title, paragraphs).encode("utf-8")
    chunks = [text[i : i + _RECORD_SIZE] for i in range(0, len(text), _RECORD_SIZE)]
    records = [_record0(len(text), len(chunks), title, author), *chunks]
    records.append(b"\xe9\x8e\r\n")
    table_size = 78 + 8 * len(records) + 2
    offsets: list[int] = []
    position = table_size
    for record in records:
        offsets.append(position)
        position += len(record)
    palm_name = title.replace(" ", "_").encode("ascii")[:31].ljust(32, b"\0")
    database = bytearray(palm_name)
    database += struct.pack(">HHLLLLLL", 0, 0, 0, 0, 0, 0, 0, 0)
    database += b"BOOKMOBI"
    database += struct.pack(">LLH", 2 * len(records) - 1, 0, len(records))
    for index, offset in enumerate(offsets):
        database += struct.pack(">LL", offset, 2 * index)
    database += b"\0\0"
    for record in records:
        database += record
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(bytes(database))
    return path


__all__ = [
    "BOOK_AUTHOR",
    "BOOK_TITLE",
    "CHAPTERS",
    "MOBI_AUTHOR",
    "MOBI_PARAGRAPHS",
    "MOBI_TITLE",
    "chapter_html",
    "make_epub",
    "make_mobi",
    "mobi_html",
]
