import re
import ast
from typing import List, Dict, Set, Optional
import logging

logger = logging.getLogger(__name__)


def extract_code_blocks(content: str) -> List[Dict]:
    """Extract all code blocks from message content"""
    code_blocks = []

    # Pattern for fenced code blocks with optional language
    fenced_pattern = r'```(\w+)?\n(.*?)\n```'
    fenced_matches = re.findall(fenced_pattern, content, re.DOTALL)

    for language, code in fenced_matches:
        code_blocks.append({
            "type": "fenced",
            "language": language.lower() if language else "unknown",
            "code": code.strip(),
            "length": len(code.strip())
        })

    # Pattern for inline code
    inline_pattern = r'`([^`\n]+)`'
    inline_matches = re.findall(inline_pattern, content)

    for code in inline_matches:
        # Skip if it's just a single word or very short
        if len(code.strip()) > 3:
            code_blocks.append({
                "type": "inline",
                "language": "unknown",
                "code": code.strip(),
                "length": len(code.strip())
            })

    return code_blocks


def extract_imports(content: str) -> Dict[str, List[str]]:
    """Extract imports from both conversation text and code blocks"""
    imports = {
        "from_conversation": [],
        "from_code": [],
        "all_libraries": set()
    }

    # Extract from conversation text (mentions of libraries)
    conversation_imports = extract_imports_from_text(content)
    imports["from_conversation"] = conversation_imports

    # Extract from code blocks
    code_blocks = extract_code_blocks(content)
    for block in code_blocks:
        if block["language"] in ["python", "py"]:
            code_imports = extract_imports_from_python_code(block["code"])
            imports["from_code"].extend(code_imports)

    # Combine all unique libraries
    all_libs = set(imports["from_conversation"] + imports["from_code"])
    imports["all_libraries"] = list(all_libs)

    return imports


def extract_imports_from_text(content: str) -> List[str]:
    """Extract library mentions from conversation text"""
    # Common Python libraries and frameworks
    libraries = [
        'fastapi', 'django', 'flask', 'pandas', 'numpy', 'scipy', 'matplotlib',
        'seaborn', 'plotly', 'sklearn', 'tensorflow', 'pytorch', 'torch',
        'keras', 'opencv', 'pillow', 'requests', 'urllib', 'asyncio', 'aiohttp',
        'sqlalchemy', 'psycopg2', 'pymongo', 'redis', 'celery', 'pytest',
        'unittest', 'pydantic', 'typer', 'click', 'streamlit', 'gradio',
        'transformers', 'huggingface', 'openai', 'anthropic', 'langchain'
    ]

    found_libraries = []
    content_lower = content.lower()

    for lib in libraries:
        if lib in content_lower:
            found_libraries.append(lib)

    return found_libraries


def extract_imports_from_python_code(code: str) -> List[str]:
    """Extract actual import statements from Python code"""
    imports = []

    try:
        # Parse the code as AST
        tree = ast.parse(code)

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(alias.name.split('.')[0])  # Get root module
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imports.append(node.module.split('.')[0])  # Get root module
    except SyntaxError:
        # If AST parsing fails, try regex fallback
        import_patterns = [
            r'import\s+([a-zA-Z_][a-zA-Z0-9_]*)',
            r'from\s+([a-zA-Z_][a-zA-Z0-9_]*)\s+import'
        ]

        for pattern in import_patterns:
            matches = re.findall(pattern, code)
            imports.extend(matches)

    return imports


def extract_function_names(content: str) -> List[str]:
    """Extract function names and class names from code blocks"""
    functions = []

    code_blocks = extract_code_blocks(content)
    for block in code_blocks:
        if block["language"] in ["python", "py"]:
            # Extract function definitions
            func_pattern = r'def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\('
            func_matches = re.findall(func_pattern, block["code"])
            functions.extend(func_matches)

            # Extract class definitions
            class_pattern = r'class\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*[\(:]'
            class_matches = re.findall(class_pattern, block["code"])
            functions.extend([f"class_{name}" for name in class_matches])

    return functions


def extract_file_info(content: str) -> Dict[str, any]:
    """Extract file paths and types mentioned in conversation"""
    file_info = {
        "file_paths": [],
        "file_types": set(),
        "has_file_discussion": False
    }

    # Common file path patterns
    file_patterns = [
        r'([a-zA-Z0-9_/\\.-]+\.py)',  # Python files
        r'([a-zA-Z0-9_/\\.-]+\.js)',  # JavaScript files
        r'([a-zA-Z0-9_/\\.-]+\.tsx?)',  # TypeScript files
        r'([a-zA-Z0-9_/\\.-]+\.html)',  # HTML files
        r'([a-zA-Z0-9_/\\.-]+\.css)',  # CSS files
        r'([a-zA-Z0-9_/\\.-]+\.json)',  # JSON files
        r'([a-zA-Z0-9_/\\.-]+\.md)',  # Markdown files
        r'([a-zA-Z0-9_/\\.-]+\.yml)',  # YAML files
        r'([a-zA-Z0-9_/\\.-]+\.yaml)',  # YAML files
        r'([a-zA-Z0-9_/\\.-]+\.txt)',  # Text files
        r'([a-zA-Z0-9_/\\.-]+\.sql)',  # SQL files
    ]

    for pattern in file_patterns:
        matches = re.findall(pattern, content)
        for match in matches:
            file_info["file_paths"].append(match)
            # Extract file extension
            ext = match.split('.')[-1].lower()
            file_info["file_types"].add(ext)

    # Check for file-related discussion keywords
    file_keywords = ['file', 'directory', 'folder', 'path', 'module', 'script']
    content_lower = content.lower()

    for keyword in file_keywords:
        if keyword in content_lower:
            file_info["has_file_discussion"] = True
            break

    file_info["file_types"] = list(file_info["file_types"])
    return file_info


def detect_task_type(content: str, header_task_type: Optional[str] = None) -> str:
    """Auto-detect task type from conversation content"""

    # If header provides task type, use it
    if header_task_type and header_task_type.lower() != "general":
        return header_task_type.lower()

    content_lower = content.lower()

    # Task detection patterns
    task_patterns = {
        "debug": [
            "error", "exception", "bug", "fix", "debug", "traceback",
            "not working", "fails", "crash", "issue"
        ],
        "explain": [
            "explain", "understand", "how does", "what is", "what does",
            "can you tell me", "help me understand", "clarify"
        ],
        "implement": [
            "create", "build", "make", "implement", "write", "develop",
            "add feature", "new function", "generate"
        ],
        "review": [
            "review", "check", "look at", "feedback", "improve", "better way",
            "optimize", "performance"
        ],
        "refactor": [
            "refactor", "clean up", "restructure", "reorganize", "simplify",
            "better structure"
        ]
    }

    # Score each task type
    task_scores = {}
    for task_type, keywords in task_patterns.items():
        score = sum(1 for keyword in keywords if keyword in content_lower)
        if score > 0:
            task_scores[task_type] = score

    # Return the highest scoring task type, or 'general' if none found
    if task_scores:
        return max(task_scores, key=task_scores.get)

    return "general"


def is_error_discussion(content: str) -> bool:
    """Check if the conversation is related to errors or debugging"""

    # Python traceback patterns
    traceback_patterns = [
        r'Traceback \(most recent call last\):',
        r'File ".*", line \d+',
        r'\w+Error:',
        r'\w+Exception:',
    ]

    for pattern in traceback_patterns:
        if re.search(pattern, content):
            return True

    # Error keywords
    error_keywords = [
        "error", "exception", "traceback", "fails", "failed", "crash",
        "not working", "broken", "issue", "problem", "bug"
    ]

    content_lower = content.lower()
    error_count = sum(1 for keyword in error_keywords if keyword in content_lower)

    # HTTP status codes (FastAPI related)
    http_error_patterns = [
        r'[45]\d{2}',  # 4xx and 5xx status codes
        r'status.*code.*[45]\d{2}',
        r'HTTP.*[45]\d{2}'
    ]

    for pattern in http_error_patterns:
        if re.search(pattern, content):
            return True

    # Return True if multiple error indicators found
    return error_count >= 2


def get_programming_language(content: str, header_language: Optional[str] = None) -> str:
    """Determine the primary programming language discussed"""

    # If header provides language, use it
    if header_language:
        return header_language.lower()

    # Extract from code blocks
    code_blocks = extract_code_blocks(content)
    language_counts = {}

    for block in code_blocks:
        lang = block["language"]
        if lang != "unknown":
            language_counts[lang] = language_counts.get(lang, 0) + 1

    # Return most common language from code blocks
    if language_counts:
        return max(language_counts, key=language_counts.get)

    # Fallback: detect from imports and keywords
    content_lower = content.lower()

    language_indicators = {
        "python": ["python", "pip", "django", "flask", "fastapi", "pandas", "numpy"],
        "javascript": ["javascript", "js", "node", "npm", "react", "vue"],
        "typescript": ["typescript", "ts", "tsx"],
        "sql": ["select", "insert", "update", "delete", "from", "where"],
        "bash": ["bash", "shell", "chmod", "mkdir", "ls", "cd"]
    }

    for language, indicators in language_indicators.items():
        if any(indicator in content_lower for indicator in indicators):
            return language

    # Default to Python since that's the primary focus
    return "python"


def analyze_message_content(content: str, headers: Dict = None) -> Dict:
    """Main function to analyze message content and extract coding metadata"""

    headers = headers or {}

    analysis = {
        "language": get_programming_language(content, headers.get("language")),
        "task_type": detect_task_type(content, headers.get("task_type")),
        "code_blocks": extract_code_blocks(content),
        "imports": extract_imports(content),
        "functions": extract_function_names(content),
        "file_info": extract_file_info(content),
        "is_error_related": is_error_discussion(content),
        "has_code": len(extract_code_blocks(content)) > 0,
        "complexity_score": calculate_complexity_score(content)
    }

    return analysis


def calculate_complexity_score(content: str) -> int:
    """Calculate a simple complexity score for the conversation"""
    score = 0

    # Base score from content length
    score += min(len(content) // 100, 10)  # Max 10 points from length

    # Additional points for code presence
    code_blocks = extract_code_blocks(content)
    score += len(code_blocks) * 2

    # Points for imports and functions
    imports = extract_imports(content)
    score += len(imports["all_libraries"])

    functions = extract_function_names(content)
    score += len(functions)

    # Points for error discussion (indicates complexity)
    if is_error_discussion(content):
        score += 5

    return min(score, 100)  # Cap at 100