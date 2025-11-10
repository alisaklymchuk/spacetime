import os
import json
import time
import re
import random
import requests
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, Dict, List
from bs4 import BeautifulSoup


class ReutersScraper:
    def __init__(self, base_url: str = 'https://www.reuters.com'):
        self.base_url = base_url
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko)',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
        }
        self.session = requests.Session()
        self.session.headers.update(self.headers)

    def fetch_sitemap_page(self, year: str, month: str, day: str, page: int) -> Optional[str]:
        """
        Fetch a sitemap page HTML
        """
        url = f"{self.base_url}/sitemap/{year}-{month}/{day}/{page}/"
        print(f"Fetching: {url}")
        
        try:
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            return response.text
        except requests.exceptions.RequestException as e:
            print(f"Error fetching {url}: {e}")
            return None

    def extract_articles_from_html(self, html: str) -> List[Dict]:
        """
        Extract article data from the Fusion.contentCache JavaScript variable
        """
        articles = []
        
        # Look for Fusion.contentCache in the HTML
        # The pattern matches: Fusion.contentCache = { ... };
        pattern = r'Fusion\.contentCache\s*=\s*({.*?});'
        match = re.search(pattern, html, re.DOTALL)
        
        if match:
            try:
                # Parse the JSON
                cache_json = match.group(1)
                cache_data = json.loads(cache_json)
                
                # Navigate to articles-by-search-v2
                if 'articles-by-search-v2' in cache_data:
                    search_results = cache_data['articles-by-search-v2']
                    
                    # Get the first (and usually only) query result
                    for query_key, query_data in search_results.items():
                        if 'data' in query_data and 'result' in query_data['data']:
                            result = query_data['data']['result']
                            if result is None:
                                print("Result is None, skipping...")
                                break
                            if 'articles' in result:
                                articles = result['articles']
                                print(f"Found {len(articles)} articles in Fusion.contentCache")
                                break
                                
            except json.JSONDecodeError as e:
                print(f"Error parsing Fusion.contentCache JSON: {e}")
        
        # If no articles found, try the old method
        if not articles:
            articles = self._extract_from_script_tags(html)
        
        return articles

    def _extract_from_script_tags(self, html: str) -> List[Dict]:
        """
        Fallback method: extract from script tags with type="application/json"
        """
        articles = []
        soup = BeautifulSoup(html, 'html.parser')
        
        script_tags = soup.find_all('script', type='application/json')
        
        for script in script_tags:
            try:
                data = json.loads(script.string)
                if isinstance(data, dict):
                    extracted = self._extract_articles_recursive(data)
                    if extracted:
                        articles.extend(extracted)
            except (json.JSONDecodeError, AttributeError):
                continue
        
        return articles

    def _extract_articles_recursive(self, data: Dict, articles: List = None) -> List[Dict]:
        """
        Recursively search for article arrays in JSON data
        """
        if articles is None:
            articles = []
        
        if isinstance(data, dict):
            # Check if this dict has article-like structure
            if 'id' in data and 'canonical_url' in data and 'published_time' in data:
                articles.append(data)
            # Continue searching in nested structures
            for key, value in data.items():
                if key == 'articles' and isinstance(value, list):
                    articles.extend([item for item in value if isinstance(item, dict)])
                elif isinstance(value, (dict, list)):
                    self._extract_articles_recursive(value, articles)
        elif isinstance(data, list):
            for item in data:
                if isinstance(item, (dict, list)):
                    self._extract_articles_recursive(item, articles)
        
        return articles

    def get_total_pages(self, html: str) -> int:
        """
        Extract total number of pages from pagination info or Fusion data
        """
        # First try to get from Fusion.contentCache
        pattern = r'Fusion\.contentCache\s*=\s*({.*?});'
        match = re.search(pattern, html, re.DOTALL)
        
        if match:
            try:
                cache_json = match.group(1)
                cache_data = json.loads(cache_json)
                
                if 'articles-by-search-v2' in cache_data:
                    search_results = cache_data['articles-by-search-v2']
                    
                    for query_key, query_data in search_results.items():
                        if 'data' in query_data and 'result' in query_data['data']:
                            result = query_data['data']['result']
                            if 'pagination' in result and 'total_size' in result['pagination']:
                                total_items = result['pagination']['total_size']
                                # Assuming 10 items per page
                                return (total_items + 9) // 10
            except:
                pass
        
        # Fallback: Look for pagination text in HTML
        soup = BeautifulSoup(html, 'html.parser')
        pagination_text = soup.find('span', {'data-testid': 'SitemapFeedPaginationText'})
        if pagination_text:
            text = pagination_text.get_text()
            match = re.search(r'of (\d+)', text)
            if match:
                total_items = int(match.group(1))
                return (total_items + 9) // 10
        
        return 1

    def format_filename(self, article: Dict) -> str:
        """
        Format filename from article data
        """
        # Extract date and time from published_time
        try:
            pub_time = datetime.fromisoformat(article['published_time'].replace('Z', '+00:00'))
            timestamp = pub_time.strftime('%Y-%m-%dT%H-%M-%S')
        except:
            timestamp = datetime.now().strftime('%Y-%m-%dT%H-%M-%S')
        
        # Get kicker category (sanitize for filename)
        category = article.get('kicker', {}).get('name', 'uncategorized')
        if isinstance(category, str):
            sanitized_category = ''.join(c if c.isalnum() or c == '-' else '_' for c in category).lower()
        else:
            sanitized_category = 'uncategorized'
        
        # Get article ID
        uuid = article.get('id', 'unknown')
        
        return f"{timestamp}.{sanitized_category}.{uuid}.json"

    def get_folder_path(self, date: str) -> Path:
        """
        Get folder path for a date
        """
        return Path('reuters_data') / date

    def save_article(self, article: Dict, date: str) -> Path:
        """
        Save article to file
        """
        folder_path = self.get_folder_path(date)
        
        # Create directory if it doesn't exist
        folder_path.mkdir(parents=True, exist_ok=True)
        
        filename = self.format_filename(article)
        file_path = folder_path / filename
        
        # Save article data
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(article, f, indent=2, ensure_ascii=False)
        
        print(f"Saved: {file_path}")
        return file_path

    def scrape_date(self, date_string: str) -> int:
        """
        Scrape articles for a specific date
        """
        print(f"\n=== Scraping articles for {date_string} ===")
        
        # Parse date
        date_obj = datetime.strptime(date_string, '%Y-%m-%d')
        year = date_obj.strftime('%Y')
        month = date_obj.strftime('%m')
        day = date_obj.strftime('%d')
        
        saved_count = 0
        page = 1
        total_pages = None
        consecutive_failures = 0
        max_consecutive_failures = 3  # Stop after 3 consecutive pages with no articles

        while True:
            html = self.fetch_sitemap_page(year, month, day, page)
            
            if not html:
                print(f'No data available for page {page}')
                consecutive_failures += 1
                
                # If we have a known total_pages, continue to next page
                if total_pages and page < total_pages and consecutive_failures < max_consecutive_failures:
                    page += 1
                    print(f"Retrying with next page after brief delay...")
                    time.sleep(5)
                    continue
                else:
                    break

            # Get total pages from first page
            if total_pages is None:
                total_pages = self.get_total_pages(html)
                print(f"Total pages: {total_pages}")

            # Extract articles from HTML
            articles = self.extract_articles_from_html(html)
            
            if not articles:
                print(f'No articles found on page {page}')
                consecutive_failures += 1
                
                # If it's the first page and no articles, something is wrong
                if page == 1:
                    print("WARNING: No articles found on first page. This might indicate a structure change.")
                    print("Attempting to continue to next pages anyway...")
                
                # If we know there are more pages, continue trying
                if total_pages and page < total_pages and consecutive_failures < max_consecutive_failures:
                    page += 1
                    print(f"Continuing to next page after brief delay...")
                    time.sleep(5)
                    continue
                elif consecutive_failures >= max_consecutive_failures:
                    print(f"Stopping after {max_consecutive_failures} consecutive pages with no articles")
                    break
                else:
                    break
            
            # Reset consecutive failures counter when we find articles
            consecutive_failures = 0

            print(f"Processing page {page}/{total_pages}: found {len(articles)} articles")

            # Save each article
            for article in articles:
                try:
                    self.save_article(article, date_string)
                    saved_count += 1
                except Exception as e:
                    print(f"Error saving article {article.get('id', 'unknown')}: {e}")

            # Move to next page
            page += 1
            
            # Stop if we've processed all pages
            if total_pages and page > total_pages:
                break

            # Add a delay to avoid overwhelming the server
            delay = random.randint(10, 90)
            print(f"Waiting {delay} seconds before next page...")
            time.sleep(delay)

        print(f"\nCompleted: {saved_count} articles saved for {date_string}")
        return saved_count

    def scrape_date_range(self, start_date: str, end_date: str) -> int:
        """
        Scrape articles for a date range
        """
        start = datetime.strptime(start_date, '%Y-%m-%d')
        end = datetime.strptime(end_date, '%Y-%m-%d')
        
        dates = []
        current = start
        while current <= end:
            dates.append(current.strftime('%Y-%m-%d'))
            current += timedelta(days=1)

        print(f"Scraping {len(dates)} dates from {start_date} to {end_date}")

        total_saved = 0
        for date in dates:
            count = self.scrape_date(date)
            total_saved += count
            
            # Delay between dates
            delay = random.randint(60, 120)
            print(f"Waiting {delay} seconds before next date...")
            time.sleep(delay)

        print(f"\n=== Total articles saved: {total_saved} ===")
        return total_saved


def main():
    """
    Example usage
    """
    scraper = ReutersScraper()

    # Scrape a single date
    # scraper.scrape_date('2024-01-01')

    # Or scrape a date range
    scraper.scrape_date_range('2024-07-01', '2025-01-01')


if __name__ == '__main__':
    main()