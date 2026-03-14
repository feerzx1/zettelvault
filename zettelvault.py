import os
import argparse
import re
from typing import List, Dict, Tuple, Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Constants for folder structure
FOLDER_STRUCTURE = [
    "1. Inbox", "2. Ideation", "3. Research", "4. Development",
    "5. Production", "6. Knowledge", "7. Archive", "8. Personal",
    "9. Startup", "10. Engineering"
]

def create_folder_structure(dest_vault: str):
    """Create the new folder structure in the destination vault."""
    for folder in FOLDER_STRUCTURE:
        os.makedirs(os.path.join(dest_vault, folder), exist_ok=True)

def get_all_markdown_files(source_vault: str) -> List[str]:
    """Retrieve all markdown files from the source vault."""
    markdown_files = []
    for root, _, files in os.walk(source_vault):
        for file in files:
            if file.endswith('.md'):
                markdown_files.append(os.path.join(root, file))
    return markdown_files

def extract_and_remove_tags(content: str) -> Tuple[List[str], str]:
    """Extract tags from the content and remove them, preserving code blocks and URLs."""
    # Split the content into regular text and code blocks
    parts = re.split(r'(```[\s\S]*?```)', content)
    
    tags = []
    new_parts = []
    
    for i, part in enumerate(parts):
        if i % 2 == 0:  # This is not a code block
            # Temporarily replace URLs with placeholders
            urls = re.findall(r'(https?://\S+)', part)
            for j, url in enumerate(urls):
                part = part.replace(url, f'__URL_PLACEHOLDER_{j}__')
            
            # Extract tags
            tag_pattern = r'#([^\s#]+)'
            found_tags = re.findall(tag_pattern, part)
            tags.extend(found_tags)
            
            # Remove tags
            part = re.sub(tag_pattern, '', part)
            
            # Restore URLs
            for j, url in enumerate(urls):
                part = part.replace(f'__URL_PLACEHOLDER_{j}__', url)
        
        new_parts.append(part)
    
    # Join the parts back together
    new_content = ''.join(new_parts)
    
    return tags, new_content

def format_tag(tag: str) -> Optional[str]:
    """Format a tag string according to the requirements."""
    # Remove any existing '#' at the beginning
    tag = tag.lstrip('#')
    # Replace spaces and underscores with hyphens
    tag = re.sub(r'[\s_]+', '-', tag)
    # Remove any non-alphanumeric characters (except hyphens)
    tag = re.sub(r'[^\w-]', '', tag)
    # Ensure the tag is lowercase
    tag = tag.lower()
    # Remove any leading or trailing hyphens
    tag = tag.strip('-')
    # If the tag is empty after cleaning, return None
    return tag if tag else None

def process_file(file_path: str, source_vault: str, ignore_tags: List[str]) -> Dict:
    """
    Process a single markdown file and return its content and metadata.
    Convert the folder structure to tags and preserve existing tags.
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Extract relative path and convert folders to tags
    relative_path = os.path.relpath(file_path, source_vault)
    folders = os.path.dirname(relative_path).split(os.sep)
    folder_tags = [format_tag(folder) for folder in folders if folder]

    # Extract existing tags and remove them from content
    existing_tags, new_content = extract_and_remove_tags(content)
    existing_tags = [format_tag(tag) for tag in existing_tags]

    # Combine folder tags and existing tags, removing duplicates, None values, and ignored tags
    all_tags = [tag for tag in dict.fromkeys(folder_tags + existing_tags) if tag and tag not in ignore_tags]

    return {
        "content": new_content,
        "tags": all_tags,
        "original_filename": os.path.basename(file_path)
    }

def create_note(dest_vault: str, folder: str, title: str, content: str, tags: List[str]):
    """Create a new note in the destination vault, handling filename collisions."""
    base_filename = os.path.splitext(title)[0]
    extension = os.path.splitext(title)[1]
    counter = 0
    while True:
        if counter == 0:
            filename = f"{base_filename}{extension}"
        else:
            filename = f"{base_filename}_{counter:03d}{extension}"
        
        file_path = os.path.join(dest_vault, folder, filename)
        if not os.path.exists(file_path):
            break
        counter += 1

    # Create frontmatter with tags
    frontmatter = "---\ntags:\n"
    for tag in tags:
        frontmatter += f"  - {tag}\n"
    frontmatter += "---\n\n"

    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(frontmatter)
        f.write(content.strip())

def main(source_vault: str, dest_vault: str, ignore_tags: List[str]):
    """Main function to restructure the Obsidian vault."""
    create_folder_structure(dest_vault)
    
    markdown_files = get_all_markdown_files(source_vault)
    
    for file_path in markdown_files:
        file_metadata = process_file(file_path, source_vault, ignore_tags)
        
        # For now, we're copying all notes to the Inbox
        destination_folder = "1. Inbox"
        
        create_note(
            dest_vault,
            destination_folder,
            file_metadata["original_filename"],
            file_metadata["content"],
            file_metadata["tags"]
        )

    print("Vault restructuring complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Restructure Obsidian vault")
    parser.add_argument("source_vault", help="Path to the source Obsidian vault")
    parser.add_argument("dest_vault", help="Path to the destination Obsidian vault")
    parser.add_argument("--ignore-tags", nargs="+", default=[], help="List of tags to ignore")
    args = parser.parse_args()

    main(args.source_vault, args.dest_vault, args.ignore_tags)