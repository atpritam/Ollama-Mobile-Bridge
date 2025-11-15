"""
HTML parsing utilities for extracting clean, readable text from web pages.
Optimized for LLM consumption by removing navigation, ads, and formatting noise.
"""
import re
from typing import Optional
from bs4 import BeautifulSoup, Comment


class HTMLParser:
    """HTML parser for extracting clean text content from web pages."""

    # Tags to remove completely
    UNWANTED_TAGS = [
        'script', 'style', 'nav', 'header', 'footer', 'aside',
        'iframe', 'noscript', 'svg', 'form', 'button'
    ]

    # Class/ID patterns that indicate junk content
    JUNK_PATTERNS = [
        'nav', 'menu', 'sidebar', 'footer', 'header', 'banner',
        'ad', 'advertisement', 'cookie', 'social', 'share', 'promo'
    ]

    # Common junk text patterns
    _junk_text_pattern: re.Pattern = re.compile(
        r'\b(?:Jump to content|Main menu|move to sidebar|Navigation|Contents|Current events|'
        r'Random article|About\s+(?:us|Wikipedia)|Contact\s+us|Donate|Skip to|Toggle.*?navigation|'
        r'Sign in|Log in|Create account|Subscribe|Newsletter|Hamburger|Search|Advertisement|'
        r'Cookie\s+(?:Policy|Settings|Notice)|Privacy\s+Policy|Terms\s+(?:of\s+)?(?:Service|Use)|'
        r'All rights reserved|Copyright|Share|Tweet|Facebook|Twitter|LinkedIn)\b',
        re.IGNORECASE
    )

    _spam_sentence_starts: re.Pattern = re.compile(
        r'^\s*(?:click|download|buy|subscribe|follow|sign up|register)\b',
        re.IGNORECASE
    )

    @staticmethod
    def extract_text(html: str, max_length: int = 4000, url: Optional[str] = None) -> str:
        """
        Extract clean, readable text from HTML content with site-specific optimization.

        Args:
            html: Raw HTML content
            max_length: Maximum length of extracted text
            url: Optional URL to enable site-specific extraction

        Returns:
            Clean text extracted from HTML
        """
        if url:
            if "wikipedia.org" in url:
                text = HTMLParser._extract_wikipedia(html, max_length)
                if text:
                    return text
            elif "reddit.com" in url:
                text = HTMLParser._extract_reddit(html, max_length)
                if text:
                    return text

        # generic extraction
        return HTMLParser._extract_generic(html, max_length)

    @staticmethod
    def _extract_wikipedia(html: str, max_length: int) -> str:
        """
        Extract content specifically from Wikipedia articles using BeautifulSoup.

        Args:
            html: Raw HTML content from Wikipedia
            max_length: Maximum length of extracted text

        Returns:
            Clean text extracted from Wikipedia article
        """
        try:
            soup = BeautifulSoup(html, 'lxml')

            # Find main article content
            content = soup.find('div', class_='mw-parser-output')
            if not content:
                # Fallback to generic extraction
                return HTMLParser._extract_generic(html, max_length)

            # Remove unwanted elements
            for unwanted in content.find_all(['table', 'div'],
                                            class_=re.compile(r'(infobox|navbox|reflist|references|toc|catlinks)', re.I)):
                unwanted.decompose()

            # Remove edit buttons, sup tags (citations), and navigation
            for tag in content.find_all(['sup', 'span'], class_=re.compile(r'(mw-editsection|reference)', re.I)):
                tag.decompose()

            content_parts = []

            # headings and paragraphs
            for elem in content.find_all(['h2', 'h3', 'h4', 'p']):
                text = elem.get_text(separator=' ', strip=True)

                if elem.name.startswith('h'):
                    # heading
                    if len(text) > 3 and text.lower() not in ['see also', 'references', 'external links', 'notes']:
                        content_parts.append(f"\n{text}:")
                else:
                    # paragraph
                    if len(text) > 50:
                        # Remove citation markers [1], [2], etc.
                        text = re.sub(r'\[\d+\]', '', text)
                        content_parts.append(text)

            result = ' '.join(content_parts)
            result = re.sub(r'\s+', ' ', result).strip()

            if len(result) > max_length:
                result = HTMLParser._truncate_at_sentence(result, max_length)

            return result

        except Exception as e:
            print(f"  Wikipedia extraction error: {e}")
            return ""

    @staticmethod
    def _extract_reddit(html: str, max_length: int) -> str:
        """
        Extract content specifically from Reddit discussions using BeautifulSoup.

        Args:
            html: Raw HTML content from Reddit
            max_length: Maximum length of extracted text

        Returns:
            Clean text extracted from Reddit discussion
        """
        try:
            soup = BeautifulSoup(html, 'lxml')
            content_parts = []

            # title
            title = None
            title_elem = soup.find('h1')
            if not title_elem:
                # Try shreddit-title attribute
                shreddit_title = soup.find('shreddit-title')
                if shreddit_title and shreddit_title.get('title'):
                    title = shreddit_title.get('title')
            if not title_elem and not title:
                # Try og:title meta tag
                og_title = soup.find('meta', property='og:title')
                if og_title and og_title.get('content'):
                    title = og_title.get('content')

            if not title and title_elem:
                title = title_elem.get_text(strip=True)

            if title:
                content_parts.append(f"Post Title: {title}")

            # post body
            post_body = None
            # Try different selectors for post content
            for selector in [
                soup.find('div', class_=re.compile(r'md', re.I)),
                soup.find('shreddit-post'),
                soup.find('div', attrs={'data-test-id': 'post-content'}),
            ]:
                if selector:
                    body_text = selector.get_text(separator=' ', strip=True)
                    if len(body_text) > 50:
                        post_body = body_text[:1000]
                        break

            if post_body:
                content_parts.append(f"Post Body: {post_body}")

            # comments
            comments_found = []
            junk_keywords = {'reply', 'share', 'report', 'save', 'award', 'upvote',
                            'downvote', 'sort by', 'view discussions', 'more replies'}

            # Try different comment selectors
            comment_elems = (
                soup.find_all('div', class_=re.compile(r'Comment', re.I)) or
                soup.find_all('shreddit-comment') or
                soup.find_all('p', class_=re.compile(r'comment', re.I))
            )

            for comment_elem in comment_elems[:10]:
                comment_text = comment_elem.get_text(separator=' ', strip=True)

                # Filter out junk
                if (len(comment_text) > 60 and
                    not any(junk in comment_text.lower() for junk in junk_keywords)):
                    comments_found.append(f"Comment: {comment_text[:500]}")
                    if len(comments_found) >= 4:
                        break

            content_parts.extend(comments_found)

            # If no structured extraction worked, fall back to generic extraction
            if len(content_parts) <= 1:
                print("  Reddit: Structured extraction failed, using generic fallback")
                paragraphs = soup.find_all('p')[:15]
                for para in paragraphs:
                    para_text = para.get_text(separator=' ', strip=True)
                    if len(para_text) > 60:
                        content_parts.append(para_text[:500])

            result = '\n\n'.join(content_parts[:8])

            if len(result) > max_length:
                result = HTMLParser._truncate_at_sentence(result, max_length)

            print(f"  Reddit extraction: Found {len(content_parts)} content parts, {len(result)} chars total")
            return result

        except Exception as e:
            print(f"  Reddit extraction error: {e}")
            return ""

    @staticmethod
    def _extract_generic(html: str, max_length: int) -> str:
        """
        Generic extraction for general web pages using BeautifulSoup.

        Args:
            html: Raw HTML content
            max_length: Maximum length of extracted text

        Returns:
            Clean text extracted from HTML
        """
        try:
            soup = BeautifulSoup(html, 'lxml')

            # Remove HTML comments
            for comment in soup.find_all(string=lambda text: isinstance(text, Comment)):
                comment.extract()

            # Remove unwanted tags
            for tag in soup(HTMLParser.UNWANTED_TAGS):
                tag.decompose()

            # Remove elements with junk classes/IDs
            for pattern in HTMLParser.JUNK_PATTERNS:
                for elem in soup.find_all(class_=re.compile(pattern, re.I)):
                    elem.decompose()
                for elem in soup.find_all(id=re.compile(pattern, re.I)):
                    elem.decompose()

            # Try to find main content areas first
            main_content = None
            for selector in [
                soup.find('article'),
                soup.find('main'),
                soup.find('div', class_=re.compile(r'(content|article|post|entry)', re.I)),
                soup.find('div', id=re.compile(r'(content|article|post|entry)', re.I)),
            ]:
                if selector:
                    main_content = selector
                    break

            # If we found a main content area, use that; otherwise use the whole soup
            content_source = main_content if main_content else soup

            # Extract text with proper spacing
            text = content_source.get_text(separator=' ', strip=True)

            # Remove URLs
            text = re.sub(r'https?://\S+', '', text)

            # Remove common navigation/UI text patterns
            text = HTMLParser._junk_text_pattern.sub('', text)

            # Normalize whitespace
            text = re.sub(r'\s+', ' ', text).strip()

            # Split into sentences and filter out junk
            text = HTMLParser._filter_sentences(text)

            # Limit length at sentence boundaries
            if len(text) > max_length:
                text = HTMLParser._truncate_at_sentence(text, max_length)

            return text

        except Exception as e:
            print(f"  HTML extraction error: {e}")
            return ""

    @staticmethod
    def _filter_sentences(text: str) -> str:
        """
        Filter sentences to keep only significant content with spam detection.

        Args:
            text: Text to filter

        Returns:
            Filtered text with only significant sentences
        """
        sentences = re.split(r'[.!?]+', text)
        cleaned_sentences = []

        for sentence in sentences:
            sentence = sentence.strip()
            if (len(sentence) > 30 and
                sentence.count(' ') > 3 and
                not HTMLParser._spam_sentence_starts.match(sentence) and
                not sentence.endswith(('...', '>>>', '<<<'))):
                cleaned_sentences.append(sentence)

        return '. '.join(cleaned_sentences) + ('.' if cleaned_sentences and not cleaned_sentences[-1].endswith('.') else '')

    @staticmethod
    def _truncate_at_sentence(text: str, max_length: int) -> str:
        """
        Truncate text at a sentence boundary near the max length.

        Args:
            text: Text to truncate
            max_length: Maximum length

        Returns:
            Truncated text
        """
        text = text[:max_length]
        last_period = text.rfind('.')

        if last_period > max_length * 0.7:
            return text[:last_period + 1]
        else:
            last_space = text.rfind(' ')
            if last_space > 0:
                return text[:last_space] + '...'
            return text