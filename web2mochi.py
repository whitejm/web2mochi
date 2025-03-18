import litellm
import re
import os
import asyncio
from urllib.parse import urljoin, urlparse
from dotenv import load_dotenv
from crawl4ai import AsyncWebCrawler

# Load environment variables from .env file
load_dotenv()

def extract_main_content(markdown_text, text_llm_model):
    """Extracts only the main content from the markdown, filtering out navigation, ads, etc."""
    messages = [
        {
            "role": "user",
            "content": f"""
You are a helpful assistant that extracts only the main content from a webpage's markdown.

I have a markdown representation of a webpage that includes navigation elements, ads, footer content, and other non-essential parts. 

Extract ONLY the main content section that contains the actual article or information. Remove all:
- Navigation bars
- Sidebars
- Advertisements
- Footer sections
- Cookie notifications
- Popup elements
- Any other non-essential page elements

Keep all headings, paragraphs, lists, code blocks, and images that are part of the main content.

Here is the markdown content:

{markdown_text}

Return only the cleaned markdown with the main content. Do not include any explanations or comments in your response.
"""
        }
    ]
    try:
        response = litellm.completion(model=text_llm_model, messages=messages, temperature=0.0)
        content = response.choices[0].message.content
        
        # Remove any <think>...</think> sections that might be present in reasoning model output
        content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL)
        
        # Clean up any excessive newlines that might be left after removing sections
        content = re.sub(r'\n{3,}', '\n\n', content)
        
        return content.strip()
    except Exception as e:
        print(f"An error occurred during main content extraction: {e}")
        return markdown_text  # Return original if extraction fails

def get_image_description(image_url, vision_llm_model):
    """Generates a brief description of an image using a vision LLM."""
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Describe this image in detail, but be concise!",
                },
                {
                    "type": "image_url",
                    "image_url": {"url": image_url},
                },
            ],
        }
    ]
    try:
        response = litellm.completion(model=vision_llm_model, messages=messages)
        description = response.choices[0].message.content
        print(f"{image_url} description: {description}")
        return description
    except Exception as e:
        print(f"An error occurred during image description: {e}")
        return ""

def convert_relative_to_absolute_links(markdown_text, base_url):
    """Converts all relative links in the markdown to absolute URLs."""
    
    # Function to convert a single link
    def replace_link(match):
        link_text = match.group(1)
        link_url = match.group(2)
        
        # Check if the URL is relative (doesn't start with http:// or https://)
        if not (link_url.startswith("http://") or link_url.startswith("https://")):
            absolute_url = urljoin(base_url, link_url)
            return f"[{link_text}]({absolute_url})"
        return match.group(0)
    
    # Function to convert a single image link
    def replace_image_link(match):
        alt_text = match.group(1)
        image_url = match.group(2)
        
        # Check if the URL is relative
        if not (image_url.startswith("http://") or image_url.startswith("https://")):
            absolute_url = urljoin(base_url, image_url)
            return f"![{alt_text}]({absolute_url})"
        return match.group(0)
    
    # Replace regular markdown links [text](url)
    updated_text = re.sub(r'\[(.*?)\]\((.*?)\)', replace_link, markdown_text)
    
    # Replace markdown image links ![alt](url)
    updated_text = re.sub(r'!\[(.*?)\]\((.*?)\)', replace_image_link, updated_text)
    
    return updated_text

def split_markdown_by_images(markdown_text):
    """Splits Markdown text into segments based on image tags and extracts image URLs."""
    segments = []
    last_index = 0
    for match in re.finditer(r'!\[(.*?)\]\((.*?)\)', markdown_text):
        start, end = match.span()
        segments.append(markdown_text[last_index:start])
        last_index = end
    segments.append(markdown_text[last_index:])

    image_urls = [match.group(2) for match in re.finditer(r'!\[(.*?)\]\((.*?)\)', markdown_text)]
    return segments, image_urls

def insert_image_descriptions(markdown_text, image_urls, vision_llm_model):
    """Inserts image descriptions into the Markdown text."""
    updated_markdown = markdown_text
    image_index = 0
    for match in re.finditer(r'!\[(.*?)\]\((.*?)\)', updated_markdown):
        if image_index < len(image_urls):
            image_url = image_urls[image_index]
            description = get_image_description(image_url, vision_llm_model)
            start, end = match.span()
            # Insert the description after the image tag
            updated_markdown = (
                updated_markdown[:end]
                + f" <image_description>{description}</image_description> "
                + updated_markdown[end:]
            )
            image_index += 1 #increment counter
    return updated_markdown

def generate_mochi_cards_from_text(markdown_text, text_llm_model):
    """Generates Mochi cards from Markdown text (including image descriptions)."""
    messages = [
        {
            "role": "user",
            "content": f"""
Create simple two-sided Mochi flashcards from the provided Markdown text that cover all key concepts and terms. Each card should have a question side and an answer side.

The text may include images. If it does, there will be a description of the image within `<image_description>` tags, immediately following Markdown image tag like `![alt text](image_url)`. 

If an image is relevant to a question or answer it should include its markdown img tag.

Think about the questions and answers (flashcards) carefully. The contents should be pulled from the main section of the provided markdown. 

The questions and answer pairs (flashcards) should make sense without any other context given.

There should be one card per concept. Or roughly around one card per paragraph. Focous on the most important concepts in the text.

**Important Formatting Rules (from Mochi Documentation):**

*   Mochi cards are written in markdown
*   Use `---` on a line by itself to separate the question and answer sides of a *single* card.
*   Use `>>>` on a line by itself to separate *different* cards.

Here is a few-shot example to guide the output format (notice the last one shows how to include images):

```markdown
What is the capital of France?
---
Paris
>>>
What is the tallest mountain in the world?
---
Mount Everest
>>>
Describe the appearance of a typical house cat.
---
A typical house cat has fur, four legs, a tail, whiskers, and pointed ears. They come in a variety of colors and sizes.
>>>
What are the main components of a typical plant cell?
---
The main components of a typical plant cell are the cell wall, cell membrane, nucleus, chloroplasts, and vacuoles. ![Plant Cell Diagram](https://upload.wikimedia.org/wikipedia/commons/thumb/4/46/Plant_cell_structure_svg.svg/1920px-Plant_cell_structure_svg.svg.png)
```

Now, generate flashcards for the following content:

{markdown_text}
"""
        }
    ]
    try:
        response = litellm.completion(model=text_llm_model, messages=messages, temperature=0.0)
        return response.choices[0].message.content
    except Exception as e:
        print(f"An error occurred during flashcard generation: {e}")
        return None

async def fetch_webpage_to_markdown(url):
    """Fetches a webpage and converts it to markdown using crawl4ai."""
    try:
        async with AsyncWebCrawler() as crawler:
            result = await crawler.arun(url=url)
            return result.markdown
    except Exception as e:
        print(f"Error fetching webpage: {e}")
        return None

async def main_async():
    url = input("Enter the URL of the webpage: ")
    vision_llm_model = "groq/llama-3.2-90b-vision-preview"
    text_llm_model = "groq/deepseek-r1-distill-llama-70b-specdec"  # Or any other suitable text-only LLM

    # Set the API key for LiteLLM
    litellm.api_key = os.getenv("GROQ_API_KEY")

    # Fetch and convert the webpage to markdown
    print(f"Fetching and converting webpage: {url}")
    markdown_content = await fetch_webpage_to_markdown(url)
    
    if not markdown_content:
        print("Failed to fetch and convert webpage. Exiting.")
        return

    # Convert relative links to absolute
    base_url = url
    markdown_content = convert_relative_to_absolute_links(markdown_content, base_url)
    
    # Extract only the main content
    main_content = extract_main_content(markdown_content, text_llm_model)
    
    print(main_content)
    input("enter to continue....")
    segments, image_urls = split_markdown_by_images(main_content)
    updated_markdown = insert_image_descriptions(main_content, image_urls, vision_llm_model)
    mochi_cards = generate_mochi_cards_from_text(updated_markdown, text_llm_model)

    if mochi_cards:
        print("\nGenerated Mochi Cards:\n")
        print(mochi_cards)
        
        # Optionally save the cards to a file
        output_file = f"mochi_cards_{urlparse(url).netloc}.md"
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(mochi_cards)
        print(f"\nSaved cards to {output_file}")

def main():
    """Entry point for the program."""
    asyncio.run(main_async())

if __name__ == "__main__":
    main()