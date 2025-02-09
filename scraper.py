import re
from collections import Counter
import tokenizer as t
from urllib.parse import urlparse
from urllib.parse import urljoin
from bs4 import BeautifulSoup

VALID_DOMAINS = ("ics.uci.edu", "cs.uci.edu", "informatics.uci.edu", "stat.uci.edu")

# set of all unqiue URLs identified, might need to change to a dictionary to get urls from each page
TOTAL_URLS = set()

# AUTHORITY_COUNT = {}

SUB_DOMAIN_COUNT = {}


def scraper(url, resp):
    links = extract_next_links(url, resp)
    return [link for link in links if is_valid(link)]


def extract_next_links(url, resp):
    # Implementation required. url: the URL that was used to get the page resp.url: the actual url of the page
    # resp.status: the status code returned by the server. 200 is OK, you got the page. Other numbers mean that there
    # was some kind of problem.
    # resp.error: when status is not 200, you can check the error here, if needed.
    # resp.raw_response: this is where the page actually is. More specifically, the raw_response has two parts:
    # resp.raw_response.url: the url, again resp.raw_response.content: the content of the page! Return a list with
    # the hyperlinks (as strings) scrapped from resp.raw_response.content

    if is_valid(url) and not resp.status != 200:  # make sure the URL is valid and is not responding with error
        soup = BeautifulSoup(resp.raw_response.content, 'lxml')
        urls = parse_urls(url, soup)
        # file = open('testcontent.txt', 'w')
        # file.write(soup.getText())
        # file.close()
        # tokens = t.tokenize('testcontent.txt')
        # frequencies = t.compute_word_frequencies(tokens)
        # t.print_frequencies(frequencies)

        valid_urls = []

        for url in urls:
            valid_urls.append(url)

        return valid_urls

    return list()


def is_duplicate_url(url):
    # checks if urls are valid
    if url not in TOTAL_URLS:
        TOTAL_URLS.add(url)
        return False
    else:
        return True


def is_recursive_url(url, threshold=3):
    # Detects recursive traps where path segments are repeated too many times.
    parsed = urlparse(url)
    path_segments = parsed.path.strip("/").split("/")  # Extract path segments

    # Count occurrences of each segment
    segment_counts = Counter(path_segments)

    # If any segment appears more than the threshold, it's likely a trap
    for segment, count in segment_counts.items():
        if count >= threshold:
            return True  # Recursive pattern detected

    return False  # No recursive pattern


# def is_authority_pass_threshold(authority, threshold=500):
#     if authority in AUTHORITY_COUNT:
#         if AUTHORITY_COUNT[authority] >= threshold: # over the threshold
#             print('count = ', AUTHORITY_COUNT[authority])
#             return True
#         else:
#             AUTHORITY_COUNT[authority] += 1 # add one to occurrences
#             return False
#     else:
#         AUTHORITY_COUNT[authority] = 1 # first appearence
#         return False


def parse_urls(base_url, soup):
    urls = []
    for link in soup.find_all("a"):
        href = link.get('href')
        if href:
            full_url = urljoin(base_url, href)  # Convert to absolute URL
            defragmented_url = full_url.split('#')[0]
            urls.append(defragmented_url)
    return urls


def _is_valid_authority(url):
    # checks if url authority matches one of the allowed authorities
    parsed = urlparse(url)
    netloc = parsed.netloc.lower()  # Normalize to lowercase for consistency
    # Construct regex dynamically for allowed netlocs
    pattern = r"(" + r"|".join(re.escape(n) for n in VALID_DOMAINS) + r")$"
    return re.search(pattern, netloc) is not None


def is_valid(url):
    # Decide whether to crawl this url or not.
    # If you decide to crawl it, return True; otherwise return False.
    # There are already some conditions that return False.

    try:
        parsed = urlparse(url)
        # Check if scheme is https or http
        if parsed.scheme not in set(["http", "https"]):
            return False
        # Check if URL is one of VALID_DOMAINS
        if not _is_valid_authority(url):
            return False
        # Check if URL has already been crawled
        if is_duplicate_url(url):
            return False
        # Check if the path of the URL is recursive
        if is_recursive_url(url):
            return False

        # if is_authority_pass_threshold(parsed.netloc):
        #     print('too much authority', url)
        #     return False

        # Get rid of unwanted file types
        return not re.match(
            r".*\.(css|js|bmp|gif|jpe?g|ico"
            + r"|png|tiff?|mid|mp2|mp3|mp4"
            + r"|wav|avi|mov|mpeg|ram|m4v|mkv|ogg|ogv|pdf"
            + r"|ps|eps|tex|ppt|pptx|doc|docx|xls|xlsx|names"
            + r"|data|dat|exe|bz2|tar|msi|bin|7z|psd|dmg|iso"
            + r"|epub|dll|cnf|tgz|sha1"
            + r"|thmx|mso|arff|rtf|jar|csv"
            + r"|rm|smil|wmv|swf|wma|zip|rar|gz)$", parsed.path.lower())

    except TypeError:
        print("TypeError for ", parsed)
        raise
