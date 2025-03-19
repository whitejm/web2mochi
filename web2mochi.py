import litellm
import re
import os
import asyncio
import logging
import time
from urllib.parse import urljoin, urlparse
from dotenv import load_dotenv
from crawl4ai import AsyncWebCrawler

# Load environment variables from .env file
load_dotenv()

# Configure logging
# Set root logger to INFO to suppress DEBUG logs from external libraries
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# Get and configure our application logger specifically
logger = logging.getLogger('web2mochi')
logger.setLevel(logging.DEBUG)  # Only our logger shows DEBUG

# Suppress specific external loggers if needed
litellm_logger = logging.getLogger('LiteLLM')
litellm_logger.setLevel(logging.WARNING)  # Only show WARNING and above for LiteLLM

def extract_main_content(markdown_text, text_llm_model):
    """Extracts only the main content from the markdown, filtering out navigation, ads, etc."""
    logger.debug("Extracting main content from markdown")
    start_time = time.time()
    
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
        
        elapsed_time = time.time() - start_time
        logger.debug(f"Main content extraction completed in {elapsed_time:.2f} seconds")
        
        return content.strip()
    except Exception as e:
        logger.error(f"An error occurred during main content extraction: {e}")
        return markdown_text  # Return original if extraction fails

def get_image_description(image_url, vision_llm_model):
    """Generates a brief description of an image using a vision LLM."""
    logger.debug(f"Getting description for image: {image_url}")
    
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
        logger.debug(f"Image description received: {description[:50]}...")  # Log first 50 chars
        return description
    except Exception as e:
        logger.error(f"An error occurred during image description: {e}")
        return ""

def convert_relative_to_absolute_links(markdown_text, base_url):
    """Converts all relative links in the markdown to absolute URLs."""
    logger.debug(f"Converting relative links to absolute using base URL: {base_url}")
    
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
    
    logger.debug("Relative links conversion completed")
    return updated_text

def split_markdown_by_images(markdown_text):
    """Splits Markdown text into segments based on image tags and extracts image URLs."""
    logger.debug("Splitting markdown by images and extracting URLs")
    
    segments = []
    last_index = 0
    for match in re.finditer(r'!\[(.*?)\]\((.*?)\)', markdown_text):
        start, end = match.span()
        segments.append(markdown_text[last_index:start])
        last_index = end
    segments.append(markdown_text[last_index:])

    image_urls = [match.group(2) for match in re.finditer(r'!\[(.*?)\]\((.*?)\)', markdown_text)]
    
    logger.debug(f"Found {len(image_urls)} images in markdown")
    return segments, image_urls

def insert_image_descriptions(markdown_text, image_urls, vision_llm_model):
    """Inserts image descriptions into the Markdown text."""
    logger.debug("Inserting image descriptions into markdown")
    start_time = time.time()
    
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
    
    elapsed_time = time.time() - start_time
    logger.debug(f"Image descriptions inserted for {image_index} images in {elapsed_time:.2f} seconds")
    return updated_markdown

def generate_mochi_cards_from_text(markdown_text, text_llm_model):
    """Generates Mochi cards from Markdown text (including image descriptions)."""
    logger.info("Generating Mochi flashcards from processed text")
    start_time = time.time()
    
    messages = [
        {
            "role": "user",
            "content": f"""
Create simple two-sided Mochi flashcards from the provided Markdown text that cover all key concepts and terms. Each card should have a question side and an answer side.

The text may include images. If it does, there will be a description of the image within `<image_description>` tags, immediately following Markdown image tag like `![alt text](image_url)`. 

IMPORTANT ABOUT IMAGES: For each image, carefully consider whether it should be included:
1. In the QUESTION section (before `---`) (if the question is specifically about the image or requires the image to understand)
2. In the ANSWER section (after `---`) (if the image helps explain or visualize the answer)
3. In BOTH sections (if the image is needed for both understanding the question and seeing the answer)

Think about the questions and answers (flashcards) carefully. The contents should be pulled from the main section of the provided markdown. 

The questions and answer pairs (flashcards) should make sense without any other context given.

There should be one card per concept. Or roughly around one card per paragraph. Focus on the most important concepts in the text.

**Important Formatting Rules (from Mochi Documentation):**

*   Mochi cards are written in markdown
*   Start each card with `>>> #` followed by a blank line where `#` is the order number of the card
*   Use `---` on a line by itself to separate the question and answer sides of a *single* card.


Here is a few-shot example to guide the output format (notice how images are placed appropriately):

```markdown
>>> 1

What is the capital of France?
---
Paris

>>> 2

What is the tallest mountain in the world?
---
Mount Everest

>>> 3

Describe the appearance of a typical house cat.
---
A typical house cat has fur, four legs, a tail, whiskers, and pointed ears. They come in a variety of colors and sizes.

>>> 4

What are the main components of a typical plant cell as shown in this diagram?
![Plant Cell Diagram](https://example.com/plant_cell.png)
---
The main components of a typical plant cell are the cell wall, cell membrane, nucleus, chloroplasts, and vacuoles.
![Plant Cell Diagram](https://example.com/plant_cell.png)

>>> 5

Identify the structure labeled 'A' in this circuit diagram:
![Circuit Diagram](https://example.com/circuit.png)
---
The structure labeled 'A' is a resistor.
```

Now, generate flashcards for the following content (Return only the flashcards in raw markdown, no other response): 

{markdown_text}
"""
        }
    ]
    try:
        response = litellm.completion(model=text_llm_model, messages=messages, temperature=0.0)
        cards = response.choices[0].message.content
        
        elapsed_time = time.time() - start_time
        logger.info(f"Flashcard generation completed in {elapsed_time:.2f} seconds")
        
        # Count the number of cards generated
        card_count = cards.count('>>>')
        logger.info(f"Generated {card_count + 1} flashcards")
        
        return cards
    except Exception as e:
        logger.error(f"An error occurred during flashcard generation: {e}")
        return None

def apply_flashcard_corrections(cards, feedback, original_content, text_llm_model):
    """Applies user-requested corrections to the generated flashcards."""
    logger.info("Applying corrections to flashcards based on user feedback")
    start_time = time.time()
    
    messages = [
        {
            "role": "user",
            "content": f"""
You are a helpful assistant that modifies Mochi flashcards based on user feedback.

Original content the flashcards are based on:
```
{original_content}
```

Current flashcards:
```markdown
{cards}
```

User feedback:
{feedback}


Your task:
1. Modify the flashcards based on the user's feedback
2. Maintain the correct Mochi card format with ">>> #" numbering and "---" separators
3. Use the original content as context to create accurate cards
4. Ensure image placement is appropriate in question (berfore `---`) or answer (after `---`) or both

Return only the complete set of corrected flashcards in raw markdown (not in a markdown code block). Do not include any explanations or comments in your response.
"""
        }
    ]
    
    try:
        response = litellm.completion(model=text_llm_model, messages=messages, temperature=0.0)
        corrected_cards = response.choices[0].message.content
        
        # Remove any <think>...</think> sections that might be present in reasoning model output
        corrected_cards = re.sub(r'<think>.*?</think>', '', corrected_cards, flags=re.DOTALL)
        
        # Clean up any excessive newlines that might be left after removing sections
        corrected_cards = re.sub(r'\n{3,}', '\n\n', corrected_cards)
        
        elapsed_time = time.time() - start_time
        logger.info(f"Flashcard corrections completed in {elapsed_time:.2f} seconds")
        
        return corrected_cards
    except Exception as e:
        logger.error(f"An error occurred during flashcard correction: {e}")
        return cards  # Return original cards if correction fails

def parse_cards(mochi_cards):
    """Parses the mochi cards string into individual card entries"""
    # Clean any <think> blocks from the text
    cleaned_cards = re.sub(r'<think>.*?</think>', '', mochi_cards, flags=re.DOTALL)
    
    # Find all card blocks using regex
    # Look for patterns like ">>> 1" followed by content until the next ">>>" or end of string
    card_blocks = re.findall(r'(>>>\s*(\d+).*?)(?=(>>>\s*\d+)|$)', cleaned_cards, re.DOTALL)
    
    # Process each card block
    result = []
    for full_match, card_num, _ in card_blocks:
        # Clean the content and remove the leading ">>> X" header
        content = re.sub(r'^>>>\s*\d+\s*\n', '', full_match.strip(), flags=re.DOTALL)
        result.append((card_num.strip(), content.strip()))
    
    logger.debug(f"Parsed {len(result)} cards from the input")
    
    # Check if we parsed any cards
    if not result:
        logger.warning("No cards parsed! Using fallback parsing method")
        # Fallback: split on ">>>" and manually process
        parts = cleaned_cards.split(">>>")
        result = []
        for part in parts[1:]:  # Skip first empty part
            match = re.match(r'\s*(\d+)(.*)', part, re.DOTALL)
            if match:
                card_num, content = match.groups()
                result.append((card_num.strip(), content.strip()))
    
    return result

def save_cards_to_files(cards, url):
    """Saves individual cards to separate files in a directory structure based on the URL"""
    # Parse the URL to create directory structure
    parsed_url = urlparse(url)
    
    # Create base output directory if it doesn't exist
    base_dir = "output"
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    
    # Create a path based on the URL
    domain = parsed_url.netloc
    path = parsed_url.path.strip('/')
    
    # Create the full directory path
    dir_path = os.path.join(base_dir, domain, path)
    
    # Create directories if they don't exist
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    
    logger.info(f"Saving cards to directory: {dir_path}")
    
    # Parse the cards into individual entries
    card_entries = parse_cards(cards)
    
    # Save each card to a separate file
    for card_num, card_content in card_entries:
        filename = f"card{card_num}.md"
        file_path = os.path.join(dir_path, filename)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(card_content)
        
        logger.debug(f"Saved card {card_num} to {file_path}")
    
    # Also save the full deck for reference
    full_deck_path = os.path.join(dir_path, "full_deck.md")
    with open(full_deck_path, 'w', encoding='utf-8') as f:
        f.write(cards)
    
    logger.info(f"Saved {len(card_entries)} cards as individual files and full deck to {dir_path}")
    
    return dir_path

async def fetch_webpage_to_markdown(url):
    """Fetches a webpage and converts it to markdown using crawl4ai."""
    logger.info(f"Fetching webpage: {url}")
    start_time = time.time()
    
    try:
        async with AsyncWebCrawler() as crawler:
            result = await crawler.arun(url=url)
            
            elapsed_time = time.time() - start_time
            logger.info(f"Webpage fetched and converted to markdown in {elapsed_time:.2f} seconds")
            return result.markdown
    except Exception as e:
        logger.error(f"Error fetching webpage: {e}")
        return None

async def main_async():
    url = input("Enter the URL of the webpage: ")
    logger.info(f"Starting web2mochi processing for URL: {url}")
    
    vision_llm_model = "groq/llama-3.2-90b-vision-preview"
    text_llm_model = "groq/deepseek-r1-distill-llama-70b-specdec"  # Or any other suitable text-only LLM
    
    logger.debug(f"Using vision model: {vision_llm_model}")
    logger.debug(f"Using text model: {text_llm_model}")

    # Set the API key for LiteLLM
    litellm.api_key = os.getenv("GROQ_API_KEY")
    if not litellm.api_key:
        logger.error("GROQ_API_KEY environment variable not set")
        return

    # Fetch and convert the webpage to markdown
    markdown_content = await fetch_webpage_to_markdown(url)
    
    if not markdown_content:
        logger.error("Failed to fetch and convert webpage. Exiting.")
        return

    # Convert relative links to absolute
    base_url = url
    markdown_content = convert_relative_to_absolute_links(markdown_content, base_url)
    
    # Extract only the main content
    main_content = extract_main_content(markdown_content, text_llm_model)
    
    # Confirm step completion and continue
    logger.info("Main content extracted successfully")
    
    # Process images
    segments, image_urls = split_markdown_by_images(main_content)
    updated_markdown = insert_image_descriptions(main_content, image_urls, vision_llm_model)
    
    # Generate flashcards
    mochi_cards = generate_mochi_cards_from_text(updated_markdown, text_llm_model)

    if mochi_cards:
        logger.info("Mochi cards generated successfully")
        
        # Feedback loop for corrections
        corrections_complete = False
        current_cards = mochi_cards
        
        while not corrections_complete:
            # Display all cards for review
            card_count = current_cards.count('>>>') 
            print(f"\n--- Generated flashcards ({card_count} cards total) ---")
            print(current_cards)
            print("--- End of flashcards ---")
            
            # Ask for feedback
            print("\nReview the flashcards above. Would you like to make any corrections?")
            feedback = input("Enter your feedback (or just press Enter if no corrections needed): ")
            
            if not feedback.strip():
                logger.info("No corrections requested, finishing up")
                corrections_complete = True
                
                # Save individual card files and full deck
                output_dir = save_cards_to_files(current_cards, url)
                print(f"Flashcards saved to directory: {output_dir}")
            else:
                logger.info("Applying user-requested corrections")
                current_cards = apply_flashcard_corrections(current_cards, feedback, updated_markdown, text_llm_model)
                print("Corrections applied! Let's review again...")
    else:
        logger.error("Failed to generate Mochi cards")

def main():
    """Entry point for the program."""
    logger.info("Starting web2mochi")
    asyncio.run(main_async())
    logger.info("web2mochi completed")

if __name__ == "__main__":
    main()