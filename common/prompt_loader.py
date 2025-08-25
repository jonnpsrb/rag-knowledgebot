import os

def load_prompt_from_file(file_path="bot_prompt.md"):
    """
    Load bot prompt from a markdown file.
    Falls back to environment variable or default if file doesn't exist.
    """
    default_prompt = "You are a helpful assistant. Answer the user's questions based on the context provided."
    try:
        # Get the project root directory (where the file should be)
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(current_dir)
        prompt_file = os.path.join(project_root, file_path)
        
        if os.path.exists(prompt_file):
            with open(prompt_file, 'r', encoding='utf-8') as f:
                content = f.read()
                # Remove the markdown header if it exists
                lines = content.split('\n')
                if lines[0].startswith('# '):
                    lines = lines[1:]  # Remove header line
                    if lines and lines[0].strip() == '':
                        lines = lines[1:]  # Remove empty line after header
                return '\n'.join(lines).strip()
        else:
            # Fallback to environment variable or default
            return default_prompt
    except Exception as e:
        print(f"Error reading bot_prompt.md file: {e}")
        return default_prompt
