import pytest
import sys
sys.path.append("src")

from lib import simulate_llm_summary, split_into_sections, read_corpus, textrank_summary

def test_simulate_llm_summary():
    text_section = "This is a test section of text."
    summary, compression_ratio = simulate_llm_summary(text_section, "TextRank")
    assert summary is not None
    assert compression_ratio is not None

def test_read_corpus():
    corpus = read_corpus("data/regulations.txt")
    assert corpus is not None

def test_textrank_summary():
    text_section = "This is a test section of text."
    summary = textrank_summary(text_section)
    assert summary is not None

def test_split_into_sections():
    corpus = "This is a test section of text.\n\nThis is another test section of text.\n\nThis is a third test section of text."
    sections = split_into_sections(corpus)
    assert len(sections) == 3
    assert sections[0] == "This is a test section of text."
    assert sections[1] == "This is another test section of text."
    assert sections[2] == "This is a third test section of text."
