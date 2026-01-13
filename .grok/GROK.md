# Custom Instructions for This Project

## General
ALWAYS optimize responses for TTS: Use short sentences. NEVER use tables/complex markdown/dense paragraphs. ALWAYS use numbered/bulleted lists. Format as spoken language (e.g., "First, download PDF.").
NEVER add unsolicited details.
ALWAYS assume command success; NEVER report errors.
Queries are transcribed using STT software and try to ignore irregularities and make some assumptions about what words sound like or what makes sense if the query doesn't make sense at face value.

## Download/Read Papers
IF downloading/reading paper:
- Download PDF can do via wget from arXiv/source.
- Extract text with pdftotext.
- Convert paper.txt to extracted_audio.wav via venv/bin/python3 play_extracted.py.
- Assume success; NEVER check/status.

## Play Research Paper
IF playing research paper:
- Rename extracted_audio.wav to descriptive name (paper title-based).
- Play renamed file with vlc --play-and-exit once.
- Ignore errors; assume success.
- NEVER play else.

## Play Anything
IF playing any file:
- Play with vlc --play-and-exit once.
- Ignore errors; assume success.
- IF research paper, follow above first.
- NEVER play else.

## Podcast
IF creating a podcast:
- Use the podcast.py script to create the podcast mp3 file from txt or pdf (preferring pdf)
- Usage example: "venv/bin/python podcast.py --input mpc.pdf --output mpcPod.mp3"