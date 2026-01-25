import argparse
import json
import os
from openai import OpenAI

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Generate a long-form document using Grok API based on an input prompt.")
    parser.add_argument('--input', required=True, help="The input prompt for generating the document.")
    args = parser.parse_args()

    prompt = args.input

    # Initialize the OpenAI client with xAI's API (OpenAI-compatible)
    api_key = os.getenv("GROK_API_KEY")
    if not api_key:
        raise ValueError("GROK_API_KEY environment variable is not set.")

    client = OpenAI(base_url="https://api.x.ai/v1", api_key=api_key)

    # Step 1: Generate a structured outline
    outline_prompt = f"""
    Based on the following user prompt: '{prompt}',
    Generate a detailed outline for a long-form document (such as a detailed report, small book, short story, or movie script).
    Determine the appropriate format based on the prompt (e.g., report with sections, story with chapters, script with scenes).
    Output strictly as JSON in this format:
    {{
        "title": "Main Title of the Document",
        "sections": [
            "Section 1 Title: Brief description of what this section covers",
            "Section 2 Title: Brief description of what this section covers",
            ...
        ]
    }}
    Aim for 5-15 sections to make the document substantial.
    """

    outline_response = client.chat.completions.create(
        model="grok-4-1-fast-reasoning",
        messages=[{"role": "user", "content": outline_prompt}],
        max_tokens=1000,
        temperature=0.7,
    )

    try:
        outline_json = json.loads(outline_response.choices[0].message.content)
        title = outline_json['title']
        sections = outline_json['sections']
    except (json.JSONDecodeError, KeyError) as e:
        raise ValueError(f"Failed to parse outline JSON: {e}")

    # Step 2: Generate content for each section
    full_content = [f"# {title}\n"]

    for section in sections:
        # Split section into title and description if possible (assuming format "Title: Description")
        if ':' in section:
            section_title, section_desc = section.split(':', 1)
            section_title = section_title.strip()
            section_desc = section_desc.strip()
        else:
            section_title = section.strip()
            section_desc = ""

        section_prompt = f"""
        You are writing a section for a long-form document titled '{title}'.
        The overall document is based on this user prompt: '{prompt}'.
        This specific section is titled '{section_title}' and should cover: {section_desc}.
        Write a detailed, lengthy section (aim for 1000-2000 words if possible).
        Make it engaging, informative, and coherent with the rest of the document.
        For stories or scripts, use appropriate formatting (e.g., dialogue in scripts).
        """

        section_response = client.chat.completions.create(
            model="grok-4-1-fast-reasoning",
            messages=[{"role": "user", "content": section_prompt}],
            max_tokens=4096,  # High max_tokens to allow longer outputs; adjust based on model limits
            temperature=0.8,
        )

        section_text = section_response.choices[0].message.content.strip()
        full_content.append(f"## {section_title}\n\n{section_text}\n")

    # Step 3: Combine and save to file
    output_filename = "paper.txt"
    with open(output_filename, 'w', encoding='utf-8') as f:
        f.write("\n".join(full_content))

    print(f"Document generated and saved to '{output_filename}'.")

if __name__ == "__main__":
    main()