import re
from collections import Counter
import tokenizer as t
from urllib.parse import urlparse
from urllib.parse import urljoin
from urllib.robotparser import RobotFileParser
from bs4 import BeautifulSoup
import hashlib


VALID_DOMAINS = ("ics.uci.edu", "cs.uci.edu", "informatics.uci.edu", "stat.uci.edu")

# set of all unqiue URLs identified, might need to change to a dictionary to get urls from each page
TOTAL_URLS = set()

NORMALIZED_URL_COUNT = {}

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
    print(len(TOTAL_URLS))
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


def is_crawling_allowed(url, user_agent="*"):
    # * for all agents
    parsed = urlparse(url)
    robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt" # get url of robots.txt

    rp = RobotFileParser()

    try:
        rp.set_url(robots_url)
        rp.read()  # Read and parse robots.txt
        return rp.can_fetch(user_agent, url)  # Check if crawling is allowed
    except:
        return True  # Assume allowed if there is no robots.txt


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

def parse_urls(base_url, soup):
    urls = []
    for link in soup.find_all("a"):
        href = link.get('href')
        if href:
            full_url = urljoin(base_url, href)  # Convert to absolute URL
            defragmented_url = full_url.split('#')[0]
            urls.append(defragmented_url)
    return urls


def is_calendar_trap(url):
    parsed = urlparse(url)
    path_segments = parsed.path.strip("/").split("/")
    calendar_pattern_found = any('calendar' in segment.lower() for segment in path_segments)

    date_patterns = [
        r"\d{4}-\d{2}-\d{2}",  # YYYY-MM-DD
        r"\d{2}/\d{2}/\d{4}",  # MM/DD/YYYY
        r"\d{4}/\d{2}/\d{2}",  # YYYY/MM/DD
        r"\d{4}-\d{2}",  # YYYY-MM
    ]

    for pattern in date_patterns:
        if re.search(pattern, url):
            return True

    if calendar_pattern_found:
        return True

    return False

def _is_valid_authority(url):
    # checks if url authority matches one of the allowed authorities
    parsed = urlparse(url)
    netloc = parsed.netloc.lower()  # Normalize to lowercase for consistency
    # Construct regex dynamically for allowed netlocs
    pattern = r"(" + r"|".join(re.escape(n) for n in VALID_DOMAINS) + r")$"
    return re.search(pattern, netloc) is not None


def normalize_url(url):
    """
    Normalizes the URL by:
    - Removing the last path segment
    - Stripping query parameters and fragments
    Example:
      Input:  https://isg.ics.uci.edu/events/tag/talk/2019-11?view=full#top
      Output: https://isg.ics.uci.edu/events/tag/talk
    """
    parsed = urlparse(url)

    # Split the path into segments and remove the last one if there are multiple segments
    path_segments = parsed.path.strip("/").split("/")

    if len(path_segments) > 2:
        path_segments.pop()  # Remove the last segment

    # Reconstruct the cleaned URL (excluding query and fragment)
    cleaned_path = "/" + "/".join(path_segments)
    normalized_url = f"{parsed.scheme}://{parsed.netloc}{cleaned_path}"

    return normalized_url


def is_url_over_threshold(url, threshold=100):
    n_url = normalize_url(url)
    if n_url in NORMALIZED_URL_COUNT:
        if NORMALIZED_URL_COUNT[n_url] >= threshold: # over the threshold
            print('count = ', NORMALIZED_URL_COUNT[n_url])
            print('url = ', n_url)
            return True
        else:
            NORMALIZED_URL_COUNT[n_url] += 1 # add one to occurrences
            return False
    else:
        NORMALIZED_URL_COUNT[n_url] = 1 # first appearence
        return False


def simhash(input):
    # Split the input into a set of features
    features = extract_features(input)

    # Generate a hash for each feature
    hashes = [hashlib.sha1(feature).hexdigest() for feature in features]

    # Combine the feature hashes to produce the final simhash
    concatenated_hash = ''.join(hashes)
    simhash = hashlib.sha1(concatenated_hash).hexdigest()

    return simhash


def compare_simhashes(simhash1, simhash2):
    # Convert simhashes to integers
    int_simhash1 = int(simhash1, 16)
    int_simhash2 = int(simhash2, 16)

    # Calculate the distance between the simhashes
    distance = bin(int_simhash1 ^ int_simhash2).count('1')

    return distance


def is_same_content():
    if distance < 5:
        return True
    return False


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

        if not is_crawling_allowed(url):
            print('robots not allowing crawl', url)
            return False

        if is_url_over_threshold(url):
            print('url over threshold')
            return False

        if is_calendar_trap(url):
            print('calendar trap:', url)
            return False

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
