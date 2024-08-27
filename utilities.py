import anthropic
import os
import json
import time
from datetime import datetime
from pathlib import Path

def get_api_key():
    try:
        with open("_claude_key.txt", "r") as file:
            return file.read().strip()
    except FileNotFoundError:
        print("Error: _claude_key.txt file not found.")
        return None
    except IOError:
        print("Error: Unable to read _claude_key.txt file.")
        return None


def get_claude_response(prompt, max_retries=5):
    api_key = get_api_key()
    if not api_key:
        return None

    client = anthropic.Anthropic(api_key=api_key)

    for attempt in range(max_retries):
        try:
            response = client.messages.create(
                model="claude-3-5-sonnet-20240620",
                max_tokens=4096,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )

            response_text = response.content[0].text
            # Find the first occurrence of '{' or '['
            json_start = min(response_text.find('{'), response_text.find('['))
            if json_start != -1:
                # Extract from the first '{' or '[' to the end
                json_text = response_text[json_start:]
                return json.loads(json_text)
            else:
                raise json.JSONDecodeError("No JSON object or array found in the response", response_text, 0)

        except json.JSONDecodeError as e:
            error_message = f"Attempt {attempt + 1}: Error decoding JSON: {e}\nResponse text: {response_text}"
            print(error_message)
            save_error(error_message)

        except Exception as e:
            error_message = f"Attempt {attempt + 1}: Error occurred: {e}"
            print(error_message)
            save_error(error_message)

        if attempt < max_retries - 1:
            print(f"Retrying in 5 seconds...")
            time.sleep(5)

    error_message = f"Failed to get a valid JSON response after {max_retries} attempts."
    print(error_message)
    save_error(error_message)
    return None


def save_error(error_message):
    # Create logs/errors directory if it doesn't exist
    log_dir = os.path.join("logs", "errors")
    os.makedirs(log_dir, exist_ok=True)

    # Generate timestamp for filename
    timestamp = datetime.now().strftime("%y%m%d_%H%M%S")
    filename = f"{timestamp}.txt"

    # Full path for the error log file
    filepath = os.path.join(log_dir, filename)

    # Write error message to file
    with open(filepath, "w") as f:
        f.write(error_message)

    print(f"Error logged to {filepath}")


def save_results(results, filename, directory=None, overwrite=False, ensure_ascii=False):
    """
    Save results to a JSON file.

    Args:
    - results: The data to be saved (must be JSON serializable)
    - filename: Name of the file to save the results
    - directory: Directory to save the file (default is current directory)
    - overwrite: If True, overwrite existing file; if False, append number to filename (default False)
    - ensure_ascii: If False, allow non-ASCII characters in output (default False)

    Returns:
    - Path of the saved file
    """
    # Create full path
    if directory:
        Path(directory).mkdir(parents=True, exist_ok=True)
        full_path = Path(directory) / filename
    else:
        full_path = Path(filename)

    # Handle existing file
    if full_path.exists() and not overwrite:
        base = full_path.stem
        extension = full_path.suffix
        counter = 1
        while full_path.exists():
            full_path = full_path.with_name(f"{base}_{counter}{extension}")
            counter += 1

    # Save the file
    try:
        with open(full_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=ensure_ascii)
        print(f"Results saved successfully to {full_path}")
        return full_path
    except Exception as e:
        print(f"Error saving results: {e}")
        return None

