import asyncio
import aiohttp
import json
from urllib.parse import quote
from typing import Dict, List, Optional


class WikipediaAPIScraper:
    def __init__(self):
        self.base_url = "https://en.wikipedia.org/w/api.php"
        self.headers = {"User-Agent": "DermatologyKBBot/1.0 (furkanycy123@gmail.com)"}

    async def get_page_content(self, title: str) -> Optional[Dict]:
        """Fetch page content using Wikipedia's API"""
        params = {
            "action": "query",
            "format": "json",
            "titles": title,
            "prop": "extracts|categories",
            "explaintext": 1,  # Get plain text
            "exsectionformat": "wiki",  # Preserve section headers
            "cllimit": 500,
        }

        async with aiohttp.ClientSession(headers=self.headers) as session:
            async with session.get(self.base_url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    pages = data["query"]["pages"]
                    page = next(iter(pages.values()))

                    if "missing" in page:
                        return None

                    return {
                        "title": page.get("title", ""),
                        "content": page.get("extract", ""),
                        "categories": [
                            cat["title"] for cat in page.get("categories", [])
                        ],
                    }
                return None

    async def get_condition_list(self) -> List[str]:
        """Get the complete list of skin conditions from the main page using continuation"""
        conditions = []

        # Initial parameters
        params = {
            "action": "query",
            "format": "json",
            "titles": "List_of_skin_conditions",
            "prop": "links",
            "pllimit": 500,
        }

        async with aiohttp.ClientSession(headers=self.headers) as session:
            continue_fetching = True
            while continue_fetching:
                async with session.get(self.base_url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        pages = data["query"]["pages"]
                        page = next(iter(pages.values()))

                        if "links" in page:
                            for link in page["links"]:
                                title = link["title"]
                                if not any(
                                    x in title
                                    for x in ["List of", "Category:", "Template:"]
                                ):
                                    conditions.append(title)

                        # Check if there are more results to fetch
                        if "continue" in data:
                            # Update parameters with continue token
                            params.update(data["continue"])
                        else:
                            continue_fetching = False
                    else:
                        continue_fetching = False

                # Add a small delay between requests
                await asyncio.sleep(1)

        return conditions

    def clean_content(self, content: str) -> Dict[str, str]:
        """Organize content into sections"""
        sections = {"overview": ""}
        current_section = "overview"

        for line in content.split("\n"):
            if line.startswith("==") and line.endswith("=="):
                # New section
                section_name = line.strip("= ").lower()
                if section_name not in ["references", "external links", "see also"]:
                    current_section = section_name
                    sections[current_section] = ""
            else:
                sections[current_section] += line + "\n"

        # Clean up sections
        return {k: v.strip() for k, v in sections.items() if v.strip()}


async def main():
    scraper = WikipediaAPIScraper()

    # Get list of conditions
    print("Getting list of skin conditions...")
    conditions = await scraper.get_condition_list()
    print(f"Found {len(conditions)} conditions")

    # Process each condition
    knowledge_base = []
    for i, condition in enumerate(conditions, 1):
        print(f"Processing {i}/{len(conditions)}: {condition}")

        page_data = await scraper.get_page_content(condition)
        if page_data:
            entry = {
                "name": page_data["title"],
                "url": f"https://en.wikipedia.org/wiki/{quote(page_data['title'].replace(' ', '_'))}",
                "sections": scraper.clean_content(page_data["content"]),
                "categories": page_data["categories"],
            }
            knowledge_base.append(entry)

            # Add delay between requests
            await asyncio.sleep(1)

    # Save to file
    with open(
        "src/knowledge_base/dermatology_knowledge_base.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(knowledge_base, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    asyncio.run(main())
