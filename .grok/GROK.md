# Custom Instructions for This Project

## General
ALWAYS optimize responses for TTS: Use short sentences. NEVER use tables/complex markdown/dense paragraphs. ALWAYS use numbered/bulleted lists. Format as spoken language (e.g., "First, download PDF.").
NEVER add unsolicited details.
ALWAYS assume command success; NEVER report errors.
Queries are transcribed using STT software and try to ignore irregularities and make some assumptions about what words sound like or what makes sense if the query doesn't make sense at face value.
ALWAYS assume that if the user mentions a file without specifying the path, it is in the current working directory.Before downloading or creating files, check if the file already exists in the current directory. If it exists, use the existing file instead of downloading or creating a new one.

## Download/Read Papers
IF downloading/reading paper:
- Download PDF. Can do via wget from arXiv/other source.
- Extract text with pdftotext.
- Convert paper.txt to extracted_audio.wav via venv/bin/python3 extractAudio.py.
- Assume success; NEVER check/status.

## Play Research Paper
IF playing research paper:
- Rename extracted_audio.wav to descriptive name (paper title-based).
- Play renamed file with vlc --play-and-exit once.
- Ignore errors; assume success.

## Play Anything
IF playing any video or audio file:
- Play with vlc --play-and-exit once.
- Ignore errors; assume success.

## Podcast
IF creating a podcast:
- Use the podcast.py script to create the podcast mp3 file from txt or pdf (preferring pdf)
- Usage example: "venv/bin/python podcast.py --input mpc.pdf --output mpcPod.mp3"

## Script
IF I ask you to generate a script or text:
- Use the generateScript.py script to generate the text file based on the desired topic
- Usage: venv/bin/python generateScript.py --input 'A novel about a knight rescuing a maiden from werewolves'
- Output will be paper.txt