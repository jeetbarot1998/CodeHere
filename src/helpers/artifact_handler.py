import re
import base64
from typing import List, Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class ArtifactHandler:
    def __init__(self, size_threshold: int = 500):
        self.size_threshold = size_threshold
        self.artifact_file_types = {
            '.py', '.js', '.ts', '.tsx', '.jsx', '.json', '.html', '.css',
            '.sql', '.md', '.yaml', '.yml', '.xml', '.java', '.cpp', '.c',
            '.go', '.rs', '.php', '.rb', '.swift', '.kt', '.scala', '.sh',
            '.dockerfile', '.tf', '.hcl', '.r', '.m', '.pl', '.ps1'
        }

    def should_be_artifact(self, content: str, file_path: str = None) -> bool:
        """Determine if content should be converted to an artifact"""

        # Check size threshold
        if len(content) > self.size_threshold:
            return True

        # Check file type if file path provided
        if file_path:
            file_ext = self._get_file_extension(file_path)
            if file_ext in self.artifact_file_types:
                return True

        # Check if content looks like code (has common programming patterns)
        if self._looks_like_code(content):
            return True

        return False

    def _get_file_extension(self, file_path: str) -> str:
        """Extract file extension from path"""
        if '.' in file_path:
            return '.' + file_path.split('.')[-1].lower()
        return ''

    def _looks_like_code(self, content: str) -> bool:
        """Heuristic to detect if content looks like code"""
        code_indicators = [
            r'def\s+\w+\s*\(',  # Python functions
            r'function\s+\w+\s*\(',  # JavaScript functions
            r'class\s+\w+\s*[:\{]',  # Class definitions
            r'import\s+\w+',  # Import statements
            r'from\s+\w+\s+import',  # Python imports
            r'#include\s*<\w+>',  # C/C++ includes
            r'console\.log\s*\(',  # JavaScript console.log
            r'print\s*\(',  # Print statements
            r'SELECT\s+.*FROM',  # SQL queries
            r'<\w+.*>.*</\w+>',  # HTML/XML tags
        ]

        for pattern in code_indicators:
            if re.search(pattern, content, re.IGNORECASE):
                return True

        return False

    def extract_file_contents(self, message_content: str) -> List[Dict]:
        """Extract file contents from message that might need to be artifacts"""
        file_contents = []

        # Pattern 1: Code blocks with file paths
        code_block_pattern = r'```(\w+)?\s*([^\n]*)\n(.*?)\n```'
        code_matches = re.findall(code_block_pattern, message_content, re.DOTALL)

        for language, file_info, code in code_matches:
            # Check if file_info contains a file path
            file_path = self._extract_file_path(file_info)

            if self.should_be_artifact(code, file_path):
                file_contents.append({
                    'type': 'code_block',
                    'language': language or self._detect_language(code, file_path),
                    'file_path': file_path,
                    'content': code.strip(),
                    'size': len(code.strip())
                })

        # Pattern 2: File mentions with content (common in IDE integrations)
        file_mention_pattern = r'File:\s*([^\n]+)\n(.*?)(?=\n\nFile:|\n\n[A-Z]|\Z)'
        file_matches = re.findall(file_mention_pattern, message_content, re.DOTALL)

        for file_path, content in file_matches:
            content = content.strip()
            if self.should_be_artifact(content, file_path):
                file_contents.append({
                    'type': 'file_content',
                    'language': self._detect_language(content, file_path),
                    'file_path': file_path.strip(),
                    'content': content,
                    'size': len(content)
                })

        return file_contents

    def _extract_file_path(self, file_info: str) -> Optional[str]:
        """Extract file path from code block info line"""
        if not file_info:
            return None

        # Common patterns for file paths in code blocks
        file_patterns = [
            r'([a-zA-Z0-9_/\\.-]+\.[a-zA-Z0-9]+)',  # Basic file path with extension
            r'File:\s*([^\s]+)',  # "File: path/to/file.py"
            r'([^/\s]+/[^/\s]+\.[a-zA-Z0-9]+)',  # Relative paths
        ]

        for pattern in file_patterns:
            match = re.search(pattern, file_info)
            if match:
                return match.group(1)

        return None

    def _detect_language(self, content: str, file_path: str = None) -> str:
        """Detect programming language from content or file path"""

        # First try file extension
        if file_path:
            ext = self._get_file_extension(file_path)
            language_map = {
                '.py': 'python',
                '.js': 'javascript',
                '.ts': 'typescript',
                '.tsx': 'typescript',
                '.jsx': 'javascript',
                '.json': 'json',
                '.html': 'html',
                '.css': 'css',
                '.sql': 'sql',
                '.md': 'markdown',
                '.yaml': 'yaml',
                '.yml': 'yaml',
                '.xml': 'xml',
                '.java': 'java',
                '.cpp': 'cpp',
                '.c': 'c',
                '.go': 'go',
                '.rs': 'rust',
                '.php': 'php',
                '.rb': 'ruby',
                '.swift': 'swift',
                '.kt': 'kotlin',
                '.scala': 'scala',
                '.sh': 'bash',
            }
            if ext in language_map:
                return language_map[ext]

        # Fallback to content analysis
        if 'def ' in content and 'import ' in content:
            return 'python'
        elif 'function ' in content and ('{' in content or '=>' in content):
            return 'javascript'
        elif 'SELECT ' in content.upper() and 'FROM ' in content.upper():
            return 'sql'
        elif '<html' in content.lower() or '<!doctype' in content.lower():
            return 'html'
        elif 'class ' in content and '{' in content:
            return 'java'

        return 'text'

    def create_artifacts_for_claude(self, file_contents: List[Dict]) -> List[Dict]:
        """Convert file contents to Claude artifact format"""
        artifacts = []

        for i, file_data in enumerate(file_contents):
            artifact_id = f"file_artifact_{i + 1}"

            # Create Claude-style artifact
            artifact = {
                "type": "text",
                "text": f"""<artifacts>
                <artifact identifier="{artifact_id}" type="application/vnd.ant.code" language="{file_data['language']}" title="{file_data.get('file_path', 'Code File')}">
                {file_data['content']}
                </artifact>
                </artifacts>
                
                File: {file_data.get('file_path', 'Unknown')}
                Language: {file_data['language']}
                Size: {file_data['size']} characters
                
                This file has been loaded as an artifact for efficient processing."""
            }

            artifacts.append(artifact)

        return artifacts

    def process_message_for_artifacts(self, message_content: str) -> Tuple[str, List[Dict]]:
        """Process message and extract artifacts, return cleaned message and artifacts"""

        # Extract file contents that should be artifacts
        file_contents = self.extract_file_contents(message_content)

        if not file_contents:
            return message_content, []

        # Create artifacts
        artifacts = self.create_artifacts_for_claude(file_contents)

        # Clean the original message by removing large code blocks
        cleaned_message = self._remove_large_code_blocks(message_content, file_contents)

        # Add artifact references to the cleaned message
        artifact_refs = []
        for i, file_data in enumerate(file_contents):
            artifact_refs.append(
                f"📎 {file_data.get('file_path', 'File')} ({file_data['size']} chars) - Loaded as artifact")

        if artifact_refs:
            cleaned_message += f"\n\n**Files loaded as artifacts:**\n" + "\n".join(artifact_refs)

        logger.info(f"Created {len(artifacts)} artifacts from message content")

        return cleaned_message, artifacts

    def _remove_large_code_blocks(self, message_content: str, file_contents: List[Dict]) -> str:
        """Remove large code blocks that have been converted to artifacts"""
        cleaned = message_content

        for file_data in file_contents:
            if file_data['type'] == 'code_block':
                # Remove the entire code block if it's large
                pattern = r'```\w*\s*[^\n]*\n' + re.escape(file_data['content']) + r'\n```'
                cleaned = re.sub(pattern, f"[Code from {file_data.get('file_path', 'file')} loaded as artifact]",
                                 cleaned)

        return cleaned